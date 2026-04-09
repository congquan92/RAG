import { useState, useMemo, useCallback, memo } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import { AnimatePresence } from "framer-motion";
import { ArrowLeft, FileText, Pencil, Check, X, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UploadZone } from "./UploadZone";
import { StatsBar } from "./StatsBar";
import { DocumentFilters, type FilterStatus } from "./DocumentFilters";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";
import { CustomMetadataInput } from "./CustomMetadataInput";
import { DocumentCard } from "./DocumentCard";
import { cn } from "@/lib/utils";
import type { Document, RAGStats, DocumentStatus, KnowledgeBase, UpdateWorkspace } from "@/types";

const PROCESSING_STATUSES = new Set<DocumentStatus>(["pending", "parsing", "indexing", "processing"]);
const BATCH_PROCESS_SUPPORTED = false;

export interface PendingUploadItem {
    id: string;
    filename: string;
    file_size: number;
    phase: "uploading" | "pending" | "parsing" | "indexing" | "failed";
    chunksProcessed?: number;
    documentId?: string;
    error?: string | null;
}

function getPendingPhaseLabel(item: PendingUploadItem): string {
    if (item.phase === "uploading") return "Đang tải lên";
    if (item.phase === "pending") return "Đã nhận, chờ xử lý";
    if (item.phase === "parsing") return "Đang parse";
    if (item.phase === "indexing") return item.chunksProcessed && item.chunksProcessed > 0 ? `Đang index (${item.chunksProcessed} chunks)` : "Đang index";
    return "Thất bại";
}

interface DataPanelProps {
    workspace: KnowledgeBase | undefined;
    documents: Document[] | undefined;
    pendingUploads: PendingUploadItem[];
    docsLoading: boolean;
    ragStats: RAGStats | undefined;
    selectedDocId: string | null;
    onSelectDoc: (doc: Document) => void;
    onUpload: (file: File, customMetadata?: { key: string; value: string }[]) => void;
    isUploading: boolean;
    onDelete: (id: string) => void;
    onUpdateWorkspace: (data: UpdateWorkspace) => Promise<void>;
}

export const DataPanel = memo(function DataPanel({ workspace, documents, pendingUploads, docsLoading, ragStats, selectedDocId, onSelectDoc, onUpload, isUploading, onDelete, onUpdateWorkspace }: DataPanelProps) {
    const navigate = useNavigate();
    const [deleteDocConfirm, setDeleteDocConfirm] = useState<string | null>(null);
    const [searchQuery, setSearchQuery] = useState("");
    const [statusFilter, setStatusFilter] = useState<FilterStatus>("all");
    const [isEditingName, setIsEditingName] = useState(false);
    const [editName, setEditName] = useState("");
    const [editDesc, setEditDesc] = useState("");
    const [customMetadata, setCustomMetadata] = useState<{ key: string; value: string }[]>([]);

    const handleUpload = useCallback(
        (file: File) => {
            const validMeta = customMetadata.filter((m) => m.key.trim() !== "");
            onUpload(file, validMeta.length > 0 ? validMeta : undefined);
            // Optional: clear metadata after successful upload? Leaving it for convenience if they upload multiple.
        },
        [customMetadata, onUpload],
    );

    const processingCount = useMemo(() => {
        const fromDocuments = documents?.filter((d) => PROCESSING_STATUSES.has(d.status)).length ?? 0;
        const fromPendingUploads = pendingUploads.filter((u) => u.phase !== "failed").length;
        return fromDocuments + fromPendingUploads;
    }, [documents, pendingUploads]);

    const filteredDocs = useMemo(() => {
        if (!documents) return [];
        let result = documents;
        if (statusFilter !== "all") {
            if (statusFilter === "parsing") {
                result = result.filter((d) => PROCESSING_STATUSES.has(d.status));
            } else {
                result = result.filter((d) => d.status === statusFilter);
            }
        }
        if (searchQuery.trim()) {
            const q = searchQuery.toLowerCase();
            result = result.filter((d) => d.original_filename.toLowerCase().includes(q));
        }
        return result;
    }, [documents, statusFilter, searchQuery]);

    const statusCounts = useMemo(() => {
        const counts: Record<string, number> = { all: 0 };
        documents?.forEach((d) => {
            counts.all = (counts.all || 0) + 1;
            counts[d.status] = (counts[d.status] || 0) + 1;
        });
        return counts as Record<FilterStatus, number>;
    }, [documents]);

    const handleStartEdit = () => {
        if (workspace) {
            setEditName(workspace.name);
            setEditDesc(workspace.description || "");
            setIsEditingName(true);
        }
    };

    const handleSaveEdit = async () => {
        if (!editName.trim()) return;
        await onUpdateWorkspace({
            name: editName.trim(),
            description: editDesc.trim() || undefined,
        });
        setIsEditingName(false);
    };

    return (
        <div className="h-full flex flex-col border-r overflow-hidden">
            {/* Header — workspace name */}
            <div className="flex-shrink-0 px-3 pt-3 pb-2 border-b space-y-1.5">
                <button onClick={() => navigate("/")} className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-foreground transition-colors">
                    <ArrowLeft className="w-3 h-3" />
                    Trang chủ
                </button>

                {isEditingName ? (
                    <div className="space-y-1.5">
                        <Input value={editName} onChange={(e) => setEditName(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSaveEdit()} placeholder="Tên" autoFocus className="text-sm font-semibold h-8" />
                        <Input value={editDesc} onChange={(e) => setEditDesc(e.target.value)} placeholder="Mô tả" className="text-xs h-7" />
                        <div className="flex items-center gap-1">
                            <Button size="sm" onClick={handleSaveEdit} disabled={!editName.trim()} className="h-6 text-[10px] px-2">
                                <Check className="w-3 h-3 mr-0.5" /> Lưu
                            </Button>
                            <Button size="sm" variant="ghost" onClick={() => setIsEditingName(false)} className="h-6 text-[10px] px-2">
                                <X className="w-3 h-3 mr-0.5" /> Hủy
                            </Button>
                        </div>
                    </div>
                ) : (
                    <div className="flex items-center gap-1.5">
                        <div className="flex-1 min-w-0">
                            <h1 className="text-sm font-bold truncate">{workspace?.name || "Knowledge Base"}</h1>
                            {workspace?.description && <p className="text-[10px] text-muted-foreground truncate">{workspace.description}</p>}
                        </div>
                        <Button size="icon" variant="ghost" onClick={handleStartEdit} className="h-6 w-6 flex-shrink-0">
                            <Pencil className="w-3 h-3" />
                        </Button>
                    </div>
                )}
            </div>

            {/* Upload zone header & settings */}
            <div className="flex-shrink-0 px-3 py-1.5 flex items-center justify-between border-t border-b">
                <h3 className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider">Thêm tài liệu</h3>
                <CustomMetadataInput metadata={customMetadata} onChange={setCustomMetadata} />
            </div>

            {/* Upload zone — always visible, ~15% */}
            <div className="flex-shrink-0 px-3 pt-2 pb-1" style={{ height: "15%" }}>
                <UploadZone onUpload={handleUpload} isUploading={isUploading} mini />
            </div>

            {/* Stats bar */}
            <div className="flex-shrink-0 px-3 py-1.5 border-b space-y-1.5">
                <div className="flex items-center justify-between">
                    <h2 className="text-xs font-semibold flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5" />
                        Tài liệu
                    </h2>
                    <span className="text-[10px] text-muted-foreground">{documents?.length ?? 0} tệp</span>
                </div>
                <StatsBar stats={ragStats} processingCount={processingCount} />

                {/* Analyze All banner — hidden when server does not support batch processing */}
                {BATCH_PROCESS_SUPPORTED && (
                    <button
                        onClick={() => toast.info("Tạm thời chưa hỗ trợ xử lý hàng loạt thủ công.")}
                        disabled
                        className={cn("w-full flex items-center justify-between gap-2 px-2.5 py-2 rounded-md", "border border-blue-400/20 bg-blue-400/[0.06]", "hover:bg-blue-400/10 transition-colors", "opacity-50 pointer-events-none")}
                    >
                        <div className="flex items-center gap-2 min-w-0">
                            <Sparkles className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
                            <span className="text-[11px] font-medium text-blue-400 truncate">Phân tích tất cả</span>
                        </div>
                        <span className="text-[10px] text-muted-foreground flex-shrink-0">Chưa khả dụng</span>
                    </button>
                )}
            </div>

            {/* Document list — ~80% */}
            <div className="flex-1 overflow-hidden flex flex-col">
                {pendingUploads.length > 0 && (
                    <div className="px-3 pt-2 pb-1.5 border-b">
                        <div className="rounded-md border border-amber-400/20 bg-amber-400/[0.06] px-2.5 py-2 space-y-1.5">
                            <p className="text-[11px] font-medium text-amber-400">Tệp đang xử lý</p>
                            <div className="space-y-1">
                                {pendingUploads.slice(0, 3).map((item) => (
                                    <div key={item.id} className="flex items-center justify-between gap-2 text-[11px]">
                                        <div className="min-w-0 flex items-center gap-1.5">
                                            <Loader2 className={cn("w-3 h-3 flex-shrink-0", item.phase !== "failed" && "animate-spin", item.phase === "failed" ? "text-destructive" : "text-amber-400")} />
                                            <span className="truncate text-foreground/90">{item.filename}</span>
                                        </div>
                                        <span className={cn("flex-shrink-0", item.phase === "failed" ? "text-destructive" : "text-amber-400")}>{getPendingPhaseLabel(item)}</span>
                                    </div>
                                ))}
                                {pendingUploads.length > 3 && <p className="text-[10px] text-muted-foreground">+{pendingUploads.length - 3} tệp khác</p>}
                            </div>
                        </div>
                    </div>
                )}

                {docsLoading ? (
                    <div className="flex items-center justify-center py-8">
                        <Loader2 className="w-4 h-4 animate-spin text-muted-foreground mr-2" />
                        <span className="text-xs text-muted-foreground">Đang tải...</span>
                    </div>
                ) : !documents || documents.length === 0 ? (
                    <div className="flex-1 flex items-center justify-center px-3">
                        {pendingUploads.length > 0 ? (
                            <p className="text-xs text-muted-foreground text-center">Tệp đã được tải lên và đang xử lý. Tài liệu sẽ xuất hiện trong danh sách sau ít giây.</p>
                        ) : (
                            <p className="text-xs text-muted-foreground text-center">Chưa có tài liệu. Hãy kéo thả tệp ở phía trên để bắt đầu.</p>
                        )}
                    </div>
                ) : (
                    <>
                        <div className="px-3 pt-2 flex-shrink-0">
                            <DocumentFilters searchQuery={searchQuery} onSearchChange={setSearchQuery} statusFilter={statusFilter} onStatusChange={setStatusFilter} counts={statusCounts} />
                        </div>

                        <div className="flex-1 overflow-y-auto px-3 py-2 space-y-1.5">
                            <AnimatePresence mode="popLayout">
                                {filteredDocs.map((doc) => (
                                    <DocumentCard key={doc.id} doc={doc} selected={doc.id === selectedDocId} onDelete={setDeleteDocConfirm} onClick={onSelectDoc} />
                                ))}
                            </AnimatePresence>
                            {filteredDocs.length === 0 && documents.length > 0 && <div className="text-center py-4 text-[11px] text-muted-foreground">Không có tài liệu khớp bộ lọc</div>}
                        </div>
                    </>
                )}
            </div>

            {/* Delete confirmation */}
            <ConfirmDialog
                open={deleteDocConfirm !== null}
                onConfirm={async () => {
                    if (deleteDocConfirm !== null) {
                        onDelete(deleteDocConfirm);
                        setDeleteDocConfirm(null);
                    }
                }}
                onCancel={() => setDeleteDocConfirm(null)}
                title="Xóa tài liệu"
                message="Bạn có chắc chắn muốn xóa? Hành động này sẽ xóa tài liệu và dữ liệu đã index của tài liệu đó."
                confirmLabel="Xóa"
                variant="danger"
            />
        </div>
    );
});
