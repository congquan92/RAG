import { memo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { SearchX, Inbox } from "lucide-react";
import { ResultCard } from "./ResultCard";
import { KGSummary } from "./KGSummary";
import { ImageResultGrid } from "./ImageResultGrid";
import type { RAGQueryResponse } from "@/types";

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------
function ResultSkeleton() {
    return (
        <div className="space-y-3">
            {[1, 2, 3].map((i) => (
                <div key={i} className="rounded-lg border bg-card/40 px-4 py-3 space-y-2 animate-pulse">
                    <div className="flex items-center gap-2">
                        <div className="w-5 h-5 rounded-full bg-muted" />
                        <div className="h-3 bg-muted rounded w-32" />
                        <div className="h-3 bg-muted rounded w-16 ml-auto" />
                    </div>
                    <div className="space-y-1.5">
                        <div className="h-3 bg-muted rounded w-full" />
                        <div className="h-3 bg-muted rounded w-4/5" />
                        <div className="h-3 bg-muted rounded w-3/5" />
                    </div>
                    <div className="h-1 bg-muted rounded-full w-full" />
                </div>
            ))}
        </div>
    );
}

// ---------------------------------------------------------------------------
// Empty states
// ---------------------------------------------------------------------------
function NoResults({ query }: { query: string }) {
    return (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col items-center py-10 text-center">
            <SearchX className="w-10 h-10 text-muted-foreground/40 mb-3" />
            <p className="text-sm font-medium">Không tìm thấy nội dung liên quan</p>
            <p className="text-xs text-muted-foreground mt-1 max-w-xs">Thử diễn đạt lại câu hỏi hoặc kiểm tra tài liệu đã được index. Truy vấn: &quot;{query}&quot;</p>
        </motion.div>
    );
}

function SearchPrompt() {
    return (
        <div className="flex flex-col items-center py-10 text-center">
            <Inbox className="w-10 h-10 text-muted-foreground/30 mb-3" />
            <p className="text-sm text-muted-foreground">Đặt câu hỏi để tìm trên các tài liệu đã index</p>
        </div>
    );
}

// ---------------------------------------------------------------------------
// SearchResults
// ---------------------------------------------------------------------------
interface SearchResultsProps {
    results: RAGQueryResponse | null;
    isSearching: boolean;
    hasSearched: boolean;
    query: string;
}

export const SearchResults = memo(function SearchResults({ results, isSearching, hasSearched, query }: SearchResultsProps) {
    if (isSearching) return <ResultSkeleton />;

    if (!hasSearched) return <SearchPrompt />;

    if (results && results.chunks.length === 0) return <NoResults query={query} />;

    if (!results) return null;

    return (
        <div className="space-y-4">
            {/* Summary header */}
            <p className="text-xs text-muted-foreground">
                Tìm thấy {results.total_chunks} chunk liên quan cho &quot;{results.query}&quot;
            </p>

            {/* Knowledge Graph summary */}
            {results.knowledge_graph_summary && <KGSummary summary={results.knowledge_graph_summary} />}

            {/* Chunk results */}
            <AnimatePresence mode="popLayout">
                <div className="space-y-2">
                    {results.chunks.map((chunk, i) => (
                        <ResultCard key={chunk.chunk_id} chunk={chunk} index={i} query={query} />
                    ))}
                </div>
            </AnimatePresence>

            {/* Image results */}
            {results.image_refs && results.image_refs.length > 0 && <ImageResultGrid images={results.image_refs} />}
        </div>
    );
});
