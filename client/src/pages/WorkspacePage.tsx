import { useMemo, useCallback, useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { toast } from "sonner";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { DataPanel } from "@/components/rag/DataPanel";
import { ChatPanel } from "@/components/rag/ChatPanel";
import { VisualPanel } from "@/components/rag/VisualPanel";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useWorkspace, useUpdateWorkspace } from "@/hooks/useWorkspaces";
import { api } from "@/lib/api";
import type { Document, RAGStats, DocumentStatus, UpdateWorkspace } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>(["pending", "processing"]);

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
    error_message?: string | null;
}

interface RuntimeTaskState {
    status: DocumentStatus;
    error: string | null;
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

function mapServerDocument(document: ServerDocumentItem, workspaceId: string, runtimeTask?: RuntimeTaskState): Document {
    return {
        id: document.id,
        workspace_id: workspaceId,
        filename: document.filename,
        original_filename: document.filename,
        file_type: getFileType(document.filename, document.mime_type),
        file_size: document.file_size,
        status: mapServerStatus(document.chunk_count, document.latest_task_status, runtimeTask),
        chunk_count: document.chunk_count,
        error_message: runtimeTask?.error ?? null,
        created_at: document.created_at,
        updated_at: document.created_at,
    };
}

export function WorkspacePage() {
    const { workspaceId } = useParams<{ workspaceId: string }>();
    const queryClient = useQueryClient();

    const [taskStateByDocId, setTaskStateByDocId] = useState<Record<string, RuntimeTaskState>>({});

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
                    const nextStatus: DocumentStatus = task.status === "completed" ? "indexed" : task.status === "failed" ? "failed" : task.status;

                    setTaskStateByDocId((prev) => ({
                        ...prev,
                        [documentId]: {
                            status: nextStatus,
                            error: task.error_message ?? null,
                        },
                    }));

                    if (task.status === "completed") {
                        queryClient.invalidateQueries({ queryKey: ["documents"] });
                        return;
                    }

                    if (task.status === "failed") {
                        queryClient.invalidateQueries({ queryKey: ["documents"] });
                        toast.error(task.error_message || "Ingestion failed");
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
                    error: "Ingestion timeout: task stayed pending too long.",
                },
            }));
            queryClient.invalidateQueries({ queryKey: ["documents"] });

            toast.warning("Ingestion is taking longer than expected. Please refresh documents.");
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
        mutationFn: ({ file, customMetadata }: { file: File; customMetadata?: { key: string; value: string }[] }) => api.uploadFile<ServerDocumentUploadResponse>("/documents/upload", file, customMetadata),
        onSuccess: (payload) => {
            setTaskStateByDocId((prev) => ({
                ...prev,
                [payload.document_id]: {
                    status: "pending",
                    error: null,
                },
            }));

            queryClient.invalidateQueries({ queryKey: ["documents"] });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            toast.success("Upload accepted. Ingestion is running in background.");

            void pollIngestionTask(payload.task_id, payload.document_id);
        },
        onError: () => toast.error("Failed to upload document"),
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
            toast.success("Document deleted");
        },
        onError: () => toast.error("Failed to delete document"),
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
            try {
                await updateWorkspace.mutateAsync({ id: wsId, data });
            } catch {
                toast.info("Workspace metadata update is temporarily unavailable on server.");
            }
        },
        [wsId, updateWorkspace],
    );

    return (
        <div className="h-full overflow-hidden grid grid-cols-[minmax(220px,20%)_minmax(300px,40%)_minmax(300px,40%)]">
            <DataPanel
                workspace={workspace}
                documents={documents}
                docsLoading={docsLoading}
                ragStats={ragStats}
                selectedDocId={selectedDoc?.id ?? null}
                onSelectDoc={handleSelectDoc}
                onUpload={(file, customMetadata) => uploadDoc.mutate({ file, customMetadata })}
                isUploading={uploadDoc.isPending}
                onDelete={(id) => deleteDoc.mutate(id)}
                onUpdateWorkspace={handleUpdateWorkspace}
            />

            <ChatPanel workspaceId={workspaceId || ""} hasIndexedDocs={hasIndexedDocs} workspace={workspace ?? null} />

            <VisualPanel workspaceId={workspaceId || ""} hasDeepragDocs={hasDeepragDocs} />
        </div>
    );
}
