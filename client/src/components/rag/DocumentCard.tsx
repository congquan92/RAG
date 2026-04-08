import { memo, useState, useEffect } from "react";
import { motion } from "framer-motion";
import { FileText, FileType, Presentation, FileCode, Hash, Trash2, CheckCircle2, XCircle, Loader2, Clock, File } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { Document, DocumentStatus } from "@/types";

// ---------------------------------------------------------------------------
// File-type icon mapping
// ---------------------------------------------------------------------------
const FILE_TYPE_CONFIG: Record<string, { icon: typeof FileText; color: string }> = {
    pdf: { icon: FileText, color: "text-red-400" },
    docx: { icon: FileType, color: "text-blue-400" },
    pptx: { icon: Presentation, color: "text-orange-400" },
    txt: { icon: FileCode, color: "text-muted-foreground" },
    md: { icon: Hash, color: "text-purple-400" },
};

function getFileConfig(fileType: string) {
    const ext = fileType.replace(".", "").toLowerCase();
    return FILE_TYPE_CONFIG[ext] ?? { icon: File, color: "text-muted-foreground" };
}

// ---------------------------------------------------------------------------
// Status badge
// ---------------------------------------------------------------------------
const STATUS_CONFIG: Record<DocumentStatus, { label: string; className: string; icon: typeof CheckCircle2 }> = {
    pending: { label: "Chờ xử lý", className: "bg-muted text-muted-foreground", icon: Clock },
    parsing: { label: "Đang parsing", className: "bg-blue-400/15 text-blue-400", icon: Loader2 },
    indexing: { label: "Đang indexing", className: "bg-amber-400/15 text-amber-400", icon: Loader2 },
    processing: { label: "Đang xử lý", className: "bg-amber-400/15 text-amber-400", icon: Loader2 },
    completed: { label: "Hoàn tất", className: "bg-emerald-500/15 text-emerald-500", icon: CheckCircle2 },
    indexed: { label: "Đã index", className: "bg-primary/15 text-primary", icon: CheckCircle2 },
    failed: { label: "Thất bại", className: "bg-destructive/15 text-destructive", icon: XCircle },
};

function StatusBadge({ status }: { status: DocumentStatus }) {
    const config = STATUS_CONFIG[status] ?? STATUS_CONFIG.pending;
    const Icon = config.icon;
    const isAnimated = status === "parsing" || status === "indexing" || status === "processing";

    return (
        <span className={cn("inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded-full", config.className)}>
            <Icon className={cn("w-3 h-3", isAnimated && "animate-spin")} />
            {config.label}
        </span>
    );
}

// ---------------------------------------------------------------------------
// Metadata chips
// ---------------------------------------------------------------------------
function MetadataChips({ doc }: { doc: Document }) {
    const chips: { label: string; value: number }[] = [];
    if (doc.page_count && doc.page_count > 0) chips.push({ label: "pages", value: doc.page_count });
    if (doc.chunk_count > 0) chips.push({ label: "chunks", value: doc.chunk_count });
    if (doc.image_count && doc.image_count > 0) chips.push({ label: "images", value: doc.image_count });
    if (doc.table_count && doc.table_count > 0) chips.push({ label: "tables", value: doc.table_count });

    if (chips.length === 0) return null;

    return (
        <div className="flex items-center gap-2 mt-1">
            {chips.map((c) => (
                <span key={c.label} className="text-xs text-muted-foreground">
                    {c.value} {c.label}
                </span>
            ))}
        </div>
    );
}

// ---------------------------------------------------------------------------
// DocumentCard
// ---------------------------------------------------------------------------
interface DocumentCardProps {
    doc: Document;
    selected?: boolean;
    onDelete: (id: string) => void;
    onClick?: (doc: Document) => void;
}

export const DocumentCard = memo(function DocumentCard({ doc, selected, onDelete, onClick }: DocumentCardProps) {
    const fileConfig = getFileConfig(doc.file_type);
    const FileIcon = fileConfig.icon;
    const sizeStr = doc.file_size >= 1024 * 1024 ? `${(doc.file_size / (1024 * 1024)).toFixed(1)} MB` : `${Math.round(doc.file_size / 1024)} KB`;

    const isActive = doc.status === "parsing" || doc.status === "indexing" || doc.status === "processing";

    // Elapsed time for active processing
    const [elapsed, setElapsed] = useState("");
    useEffect(() => {
        if (!isActive) {
            setElapsed("");
            return;
        }
        const start = new Date(doc.updated_at).getTime();
        const tick = () => {
            const sec = Math.floor((Date.now() - start) / 1000);
            if (sec < 60) setElapsed(`${sec}s`);
            else setElapsed(`${Math.floor(sec / 60)}m ${sec % 60}s`);
        };
        tick();
        const id = setInterval(tick, 1000);
        return () => clearInterval(id);
    }, [isActive, doc.updated_at]);

    return (
        <motion.div
            layout
            initial={{ opacity: 0, y: 8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            className={cn(
                "group relative rounded-lg border bg-card transition-all duration-200",
                // Active processing state — animated border glow
                isActive ? "border-blue-400/50 shadow-[0_0_12px_-3px_rgba(96,165,250,0.3)]" : "border-border hover:shadow-md hover:-translate-y-0.5",
                selected && "border-primary ring-1 ring-primary/30 shadow-sm",
                doc.status === "indexed" ? "cursor-pointer" : "cursor-default",
            )}
            onClick={() => onClick?.(doc)}
        >
            {/* Shimmer overlay for active processing */}
            {isActive && (
                <div className="absolute inset-0 rounded-lg overflow-hidden pointer-events-none">
                    <div className="absolute inset-0 -translate-x-full animate-[shimmer_2s_ease-in-out_infinite] bg-gradient-to-r from-transparent via-blue-400/[0.07] to-transparent" />
                </div>
            )}

            <div className="relative px-4 py-3 flex items-start gap-3">
                {/* File icon */}
                <div className={cn("w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5 transition-colors", isActive ? "bg-blue-400/10" : "bg-muted/50")}>
                    {isActive ? <Loader2 className="w-5 h-5 text-blue-400 animate-spin" /> : <FileIcon className={cn("w-5 h-5", fileConfig.color)} />}
                </div>

                {/* Content */}
                <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                        <p className="font-medium text-sm truncate">{doc.original_filename}</p>
                        <StatusBadge status={doc.status} />
                    </div>
                    <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-xs text-muted-foreground">{sizeStr}</span>
                        {doc.parser_version && <span className="text-xs text-muted-foreground/60">{doc.parser_version}</span>}
                        {isActive && <span className="text-xs text-blue-400/80 font-medium animate-pulse">Đang phân tích{elapsed ? ` (${elapsed})` : "..."}</span>}
                    </div>
                    <MetadataChips doc={doc} />
                    {doc.error_message && <p className="text-xs text-destructive mt-1 truncate">{doc.error_message}</p>}
                </div>

                {/* Actions */}
                <div className="flex items-center gap-1 flex-shrink-0">
                    {/* Delete — hover only */}
                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={(e) => {
                            e.stopPropagation();
                            onDelete(doc.id);
                        }}
                        className="h-7 w-7 opacity-0 group-hover:opacity-100 transition-opacity"
                    >
                        <Trash2 className="w-3.5 h-3.5 text-destructive" />
                    </Button>
                </div>
            </div>
        </motion.div>
    );
});
