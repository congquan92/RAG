import { useMemo, useCallback, useEffect, useRef } from "react";
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

const PROCESSING_STATUSES = new Set<DocumentStatus>(["parsing", "indexing", "processing"]);

export function WorkspacePage() {
    const { workspaceId } = useParams<{ workspaceId: string }>();
    const queryClient = useQueryClient();
    const wsId = workspaceId ? Number(workspaceId) : null;

    // -- Workspace data --
    const { data: workspace } = useWorkspace(wsId);
    const updateWorkspace = useUpdateWorkspace();

    // -- Store --
    const { selectedDoc, selectDoc, reset: resetStore } = useWorkspaceStore();

    // Reset store when switching between workspaces
    useEffect(() => {
        resetStore();
    }, [workspaceId, resetStore]);

    // -----------------------------------------------------------------------
    // Queries
    // -----------------------------------------------------------------------
    const { data: documents, isLoading: docsLoading } = useQuery({
        queryKey: ["documents", workspaceId],
        queryFn: () => api.get<Document[]>(`/documents/workspace/${workspaceId}`),
        enabled: !!workspaceId,
        refetchInterval: (query) => {
            const docs = query.state.data;
            if (docs?.some((d) => PROCESSING_STATUSES.has(d.status))) return 3000;
            return false;
        },
    });

    const { data: ragStats } = useQuery({
        queryKey: ["rag-stats", workspaceId],
        queryFn: () => api.get<RAGStats>(`/rag/stats/${workspaceId}`),
        enabled: !!workspaceId,
    });

    // -----------------------------------------------------------------------
    // Refresh ragStats when processing finishes
    // -----------------------------------------------------------------------
    const processingCount = useMemo(() => documents?.filter((d) => PROCESSING_STATUSES.has(d.status)).length ?? 0, [documents]);

    const prevProcessingRef = useRef(processingCount);
    useEffect(() => {
        if (prevProcessingRef.current > 0 && processingCount === 0) {
            queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
        }
        prevProcessingRef.current = processingCount;
    }, [processingCount, queryClient, workspaceId]);

    // Keep selectedDoc in sync with latest document data
    useEffect(() => {
        if (selectedDoc && documents) {
            const updated = documents.find((d) => d.id === selectedDoc.id);
            if (updated && updated.status !== selectedDoc.status) {
                selectDoc(updated);
            }
        }
    }, [documents, selectedDoc, selectDoc]);

    const hasIndexedDocs = (ragStats?.indexed_documents ?? 0) > 0;
    const hasDeepragDocs = (ragStats?.nexusrag_documents ?? 0) > 0;

    // -----------------------------------------------------------------------
    // Mutations
    // -----------------------------------------------------------------------
    const uploadDoc = useMutation({
        mutationFn: ({ file, customMetadata }: { file: File; customMetadata?: { key: string; value: string }[] }) => api.uploadFile<Document>(`/documents/upload/${workspaceId}`, file, customMetadata),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            toast.success("Tải tài liệu lên thành công");
        },
        onError: () => toast.error("Không thể tải tài liệu lên"),
    });

    const deleteDoc = useMutation({
        mutationFn: (docId: number) => api.delete(`/documents/${docId}`),
        onSuccess: (_, docId) => {
            queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            if (selectedDoc?.id === docId) selectDoc(null);
            toast.success("Đã xóa tài liệu");
        },
        onError: () => toast.error("Không thể xóa tài liệu"),
    });

    const processDoc = useMutation({
        mutationFn: (docId: number) => api.post(`/rag/process/${docId}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
            toast.info("Đang phân tích tài liệu...", {
                description: "Đang parse nội dung và xây dựng search index.",
            });
        },
        onError: (error: Error) => {
            if (error.message?.includes("already being analyzed")) {
                toast.info("Tài liệu đang được phân tích", {
                    description: "Vui lòng chờ quá trình phân tích hiện tại hoàn tất.",
                });
                // Refresh to get latest status
                queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
            } else {
                toast.error("Không thể bắt đầu phân tích");
            }
        },
    });

    const reindexDoc = useMutation({
        mutationFn: (docId: number) => api.post(`/rag/reindex/${docId}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["documents", workspaceId] });
            queryClient.invalidateQueries({ queryKey: ["rag-stats", workspaceId] });
            toast.success("Đã bắt đầu phân tích lại tài liệu");
        },
        onError: () => toast.error("Không thể phân tích lại tài liệu"),
    });

    // -----------------------------------------------------------------------
    // Handlers
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Render — 3-column layout
    // -----------------------------------------------------------------------
    return (
        <div className="h-full overflow-hidden grid grid-cols-[minmax(220px,20%)_minmax(300px,40%)_minmax(300px,40%)]">
            {/* Column 1: Data Area */}
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
                onProcess={(id) => processDoc.mutate(id)}
                onReindex={(id) => reindexDoc.mutate(id)}
                isProcessing={processDoc.isPending}
                onUpdateWorkspace={handleUpdateWorkspace}
            />

            {/* Column 2: Chat Area */}
            <ChatPanel workspaceId={workspaceId || ""} hasIndexedDocs={hasIndexedDocs} workspace={workspace ?? null} />

            {/* Column 3: Visual Area */}
            <VisualPanel workspaceId={workspaceId || ""} hasDeepragDocs={hasDeepragDocs} />
        </div>
    );
}
