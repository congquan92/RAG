import { useMemo, useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DataPanel, type PendingUploadItem } from "@/components/rag/DataPanel";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { VisualPanel } from "@/components/rag/VisualPanel";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useWorkspace, useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { api } from "@/lib/api";
import type { Document, RAGStats, DocumentStatus, UpdateWorkspace } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>(["pending", "processing", "parsing", "indexing"]);

interface ServerDocumentItem {
    id: string;
    filename: string;
    file_size: number;
    mime_type: string;
    chunk_count: number;
    created_at: string;
    latest_task_status?: "pending" | "processing" | "completed" | "failed" | null;
}

interface ServerDocumentListResponse {
    documents: ServerDocumentItem[];
    total: number;
}

interface ServerDocumentUploadResponse {
    document_id: string;
    task_id: string;
}

interface ServerIngestionTaskResponse {
    status: "pending" | "processing" | "completed" | "failed";
    chunks_processed: number;
    error_message?: string | null;
}

interface RuntimeTaskState {
    status: DocumentStatus;
    chunksProcessed: number;
    error: string | null;
}

interface UploadMutationInput {
    file: File;
    customMetadata?: { key: string; value: string }[];
    clientId: string;
}

function getFileType(filename: string, mimeType: string): string {
    const ext = filename.split(".").pop()?.toLowerCase();
    if (ext) return ext;
    const mimeFallback = mimeType.split("/").pop();
    return mimeFallback || "file";
}

function mapServerStatus(chunkCount: number, latestTaskStatus?: ServerDocumentItem["latest_task_status"], runtimeTask?: RuntimeTaskState): DocumentStatus {
    if (runtimeTask) return runtimeTask.status;

    if (latestTaskStatus === "failed") return "failed";
    if (latestTaskStatus === "pending") return "pending";
    if (latestTaskStatus === "processing") return "processing";
    if (latestTaskStatus === "completed") return chunkCount > 0 ? "indexed" : "completed";

    return chunkCount > 0 ? "indexed" : "pending";
}

function mapTaskResponseToRuntimeState(task: ServerIngestionTaskResponse): RuntimeTaskState {
    if (task.status === "failed") {
        return {
            status: "failed",
            chunksProcessed: task.chunks_processed ?? 0,
            error: task.error_message ?? null,
        };
    }

    if (task.status === "completed") {
        return {
            status: "indexed",
            chunksProcessed: task.chunks_processed ?? 0,
            error: null,
        };
    }

    if (task.status === "processing") {
        return {
            status: (task.chunks_processed ?? 0) > 0 ? "indexing" : "parsing",
            chunksProcessed: task.chunks_processed ?? 0,
            error: null,
        };
    }

    return {
        status: "pending",
        chunksProcessed: task.chunks_processed ?? 0,
        error: null,
    };
}

function mapTaskResponseToPendingPhase(task: ServerIngestionTaskResponse): PendingUploadItem["phase"] {
    if (task.status === "failed") return "failed";
    if (task.status === "pending") return "pending";
    if (task.status === "processing") return (task.chunks_processed ?? 0) > 0 ? "indexing" : "parsing";
    return "indexing";
}

function mapServerDocument(document: ServerDocumentItem, workspaceId: string, runtimeTask?: RuntimeTaskState): Document {
    const liveChunkCount = runtimeTask ? Math.max(document.chunk_count, runtimeTask.chunksProcessed ?? 0) : document.chunk_count;

    return {
        id: document.id,
        workspace_id: workspaceId,
        filename: document.filename,
        original_filename: document.filename,
        file_type: getFileType(document.filename, document.mime_type),
        file_size: document.file_size,
        status: mapServerStatus(document.chunk_count, document.latest_task_status, runtimeTask),
        chunk_count: liveChunkCount,
        error_message: runtimeTask?.error ?? null,
        created_at: document.created_at,
        updated_at: document.created_at,
    };
}

export function WorkspacePage() {
    const { workspaceId } = useParams<{ workspaceId: string }>();
    const queryClient = useQueryClient();

    const [taskStateByDocId, setTaskStateByDocId] = useState<Record<string, RuntimeTaskState>>({});
    const [pendingUploads, setPendingUploads] = useState<PendingUploadItem[]>([]);

    const wsId = workspaceId ?? null;

    const { data: workspace } = useWorkspace(wsId);
    const updateWorkspace = useUpdateWorkspace();

    const { selectedDoc, selectDoc, reset: resetStore } = useWorkspaceStore();

    useEffect(() => {
        resetStore();
    }, [workspaceId, resetStore]);

    const pollIngestionTask = useCallback(
        async (taskId: string, documentId: string) => {
            const maxAttempts = 120;

            for (let attempt = 0; attempt < maxAttempts; attempt += 1) {
                try {
                    const task = await api.get<ServerIngestionTaskResponse>(`/documents/status/${taskId}`);
                    const nextRuntime = mapTaskResponseToRuntimeState(task);

                    setTaskStateByDocId((prev) => ({
                        ...prev,
                        [documentId]: {
                            status: nextRuntime.status,
                            chunksProcessed: nextRuntime.chunksProcessed,
                            error: nextRuntime.error,
                        },
                    }));

                    setPendingUploads((prev) =>
                        prev.map((item) =>
                            item.documentId === documentId
                                ? {
                                      ...item,
                                      phase: mapTaskResponseToPendingPhase(task),
                                      chunksProcessed: task.chunks_processed ?? 0,
                                      error: task.error_message ?? null,
                                  }
                                : item,
                        ),
                    );

                    if (task.status === "completed") {
                        setPendingUploads((prev) => prev.filter((item) => item.documentId !== documentId));
                        queryClient.invalidateQueries({ queryKey: ["documents"] });
                        return;
                    }

                    if (task.status === "failed") {
                        setPendingUploads((prev) => prev.map((item) => (item.documentId === documentId ? { ...item, phase: "failed", error: task.error_message ?? "Xử lý tài liệu thất bại" } : item)));
                        queryClient.invalidateQueries({ queryKey: ["documents"] });
                        toast.error(task.error_message || "Xử lý tài liệu thất bại");
                        return;
                    }
                } catch {
                    // Keep polling on transient network errors.
                }

                await new Promise((resolve) => setTimeout(resolve, 1500));
            }

            setTaskStateByDocId((prev) => ({
                ...prev,
                [documentId]: {
                    status: "failed",
                    chunksProcessed: 0,
                    error: "Het thoi gian cho xu ly: task pending qua lau.",
                },
            }));
            setPendingUploads((prev) => prev.map((item) => (item.documentId === documentId ? { ...item, phase: "failed", error: "Hết thời gian chờ xử lý (timeout)." } : item)));
            queryClient.invalidateQueries({ queryKey: ["documents"] });

            toast.warning("Xử lý tài liệu đang lâu hơn dự kiến. Vui lòng tải lại danh sách tài liệu.");
        },
        [queryClient],
    );

    const { data: documentList, isLoading: docsLoading } = useQuery({
        queryKey: ["documents"],
        queryFn: () => api.get<ServerDocumentListResponse>("/documents?skip=0&limit=200"),
        refetchInterval: () => {
            const hasRunningTasks = Object.values(taskStateByDocId).some((task) => PROCESSING_STATUSES.has(task.status));
            return hasRunningTasks ? 2500 : false;
        },
    });

    const documents = useMemo(() => {
        if (!workspaceId) return [];
        return (documentList?.documents ?? []).map((doc) => mapServerDocument(doc, workspaceId, taskStateByDocId[doc.id]));
    }, [documentList, taskStateByDocId, workspaceId]);

    useEffect(() => {
        const knownDocIds = new Set((documentList?.documents ?? []).map((doc) => doc.id));
        if (knownDocIds.size === 0) return;
        setPendingUploads((prev) => prev.filter((item) => !(item.documentId && knownDocIds.has(item.documentId))));
    }, [documentList]);

    useEffect(() => {
        if (!pendingUploads.some((item) => item.phase === "failed")) return;
        const timer = setTimeout(() => {
            setPendingUploads((prev) => prev.filter((item) => item.phase !== "failed"));
        }, 6000);
        return () => clearTimeout(timer);
    }, [pendingUploads]);

    const ragStats = useMemo<RAGStats>(() => {
        const indexedDocuments = documents.filter((doc) => doc.status === "indexed").length;
        const totalChunks = documents.reduce((sum, doc) => sum + (doc.chunk_count || 0), 0);

        return {
            workspace_id: workspaceId || "",
            total_documents: documents.length,
            indexed_documents: indexedDocuments,
            total_chunks: totalChunks,
            image_count: 0,
            nexusrag_documents: 0,
        };
    }, [documents, workspaceId]);

    useEffect(() => {
        if (selectedDoc && documents.length > 0) {
            const updated = documents.find((d) => d.id === selectedDoc.id);
            if (updated && updated.status !== selectedDoc.status) {
                selectDoc(updated);
            }
        }
    }, [documents, selectedDoc, selectDoc]);

    const hasIndexedDocs = ragStats.indexed_documents > 0;
    const hasDeepragDocs = false;

    const uploadDoc = useMutation({
        mutationFn: ({ file, customMetadata }: UploadMutationInput) => api.uploadFile<ServerDocumentUploadResponse>("/documents/upload", file, customMetadata),
        onSuccess: (payload, variables) => {
            setPendingUploads((prev) => prev.map((item) => (item.id === variables.clientId ? { ...item, phase: "pending", documentId: payload.document_id, chunksProcessed: 0, error: null } : item)));
            setTaskStateByDocId((prev) => ({
                ...prev,
                [payload.document_id]: {
                    status: "pending",
                    chunksProcessed: 0,
                    error: null,
                },
            }));

            queryClient.invalidateQueries({ queryKey: ["documents"] });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            toast.success("Đã nhận file. Hệ thống đang xử lý ở nền.");

            void pollIngestionTask(payload.task_id, payload.document_id);
        },
        onError: (_error, variables) => {
            setPendingUploads((prev) => prev.map((item) => (item.id === variables.clientId ? { ...item, phase: "failed", error: "Tải lên thất bại" } : item)));
            toast.error(`Tải lên thất bại: ${variables.file.name}`);
        },
    });

    const deleteDoc = useMutation({
        mutationFn: (docId: string) => api.delete(`/documents/${docId}`),
        onSuccess: (_, docId) => {
            setTaskStateByDocId((prev) => {
                const { [docId]: _removed, ...rest } = prev;
                return rest;
            });

            queryClient.invalidateQueries({ queryKey: ["documents"] });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            if (selectedDoc?.id === docId) selectDoc(null);
            toast.success("Đã xóa tài liệu");
        },
        onError: () => toast.error("Không thể xóa tài liệu"),
    });

    const handleSelectDoc = useCallback(
        (doc: Document) => {
            if (doc.status !== "indexed") return;
            if (selectedDoc?.id === doc.id) {
                selectDoc(null);
            } else {
                selectDoc(doc);
            }
        },
        [selectedDoc, selectDoc],
    );

    const handleUpdateWorkspace = useCallback(
        async (data: UpdateWorkspace) => {
            if (!wsId) return;
            await updateWorkspace.mutateAsync({ id: wsId, data });
        },
        [wsId, updateWorkspace],
    );

    const handleUpload = useCallback(
        (file: File, customMetadata?: { key: string; value: string }[]) => {
            const clientId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;

            setPendingUploads((prev) => [
                ...prev,
                {
                    id: clientId,
                    filename: file.name,
                    file_size: file.size,
                    phase: "uploading",
                    chunksProcessed: 0,
                    error: null,
                },
            ]);

            uploadDoc.mutate({ file, customMetadata, clientId });
        },
        [uploadDoc],
    );

    const hasActiveUploadRequest = pendingUploads.some((item) => item.phase === "uploading");

    return (
        <div className="h-full overflow-hidden grid grid-cols-[minmax(220px,20%)_minmax(300px,40%)_minmax(300px,40%)]">
            <DataPanel
                workspace={workspace}
                documents={documents}
                pendingUploads={pendingUploads}
                docsLoading={docsLoading}
                ragStats={ragStats}
                selectedDocId={selectedDoc?.id ?? null}
                onSelectDoc={handleSelectDoc}
                onUpload={handleUpload}
                isUploading={uploadDoc.isPending || hasActiveUploadRequest}
                onDelete={(id) => deleteDoc.mutate(id)}
                onUpdateWorkspace={handleUpdateWorkspace}
            />

            <ChatPanel workspaceId={workspaceId || ""} hasIndexedDocs={hasIndexedDocs} workspace={workspace ?? null} />

            <VisualPanel workspaceId={workspaceId || ""} hasDeepragDocs={hasDeepragDocs} />
        </div>
    );
}
