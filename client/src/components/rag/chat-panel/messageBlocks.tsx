import { useCallback, useContext, useEffect, useMemo, useRef, useState, memo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Bot, User, FileText, Brain, ChevronDown, ThumbsUp, ThumbsDown, ImageIcon, Copy, ClipboardCheck, FileCode, Loader2, Sparkles } from "lucide-react";
import { toast } from "sonner";
import { cn, generateId } from "@/lib/utils";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { StreamingMarkdown } from "@/components/rag/MemoizedMarkdown";
import { ThinkingTimeline } from "@/components/rag/ThinkingTimeline";
import type { AgentStep, ChatImageRef, ChatMessage, ChatSourceChunk, ChatStreamStatus } from "@/types";
import { DebugCtx, useFindDoc } from "./context";
import { STATUS_LABELS } from "./constants";
import { MarkdownWithCitations } from "./markdownWithCitations";

type RelevanceRating = "relevant" | "partial" | "not_relevant";

function SourceRatingButtons({ sourceIndex, currentRating, onRate }: { sourceIndex: string; currentRating?: RelevanceRating; onRate: (sourceIndex: string, rating: RelevanceRating) => void }) {
    return (
        <div className="flex items-center gap-0.5 ml-auto flex-shrink-0" onClick={(event) => event.stopPropagation()}>
            <button
                onClick={(event) => {
                    event.stopPropagation();
                    onRate(sourceIndex, "relevant");
                }}
                className={cn("p-0.5 rounded transition-colors", currentRating === "relevant" ? "text-emerald-500" : "text-muted-foreground/20 hover:text-emerald-500/60")}
                title="Lien quan"
            >
                <ThumbsUp className="w-2.5 h-2.5" />
            </button>
            <button
                onClick={(event) => {
                    event.stopPropagation();
                    onRate(sourceIndex, "not_relevant");
                }}
                className={cn("p-0.5 rounded transition-colors", currentRating === "not_relevant" ? "text-destructive" : "text-muted-foreground/20 hover:text-destructive/60")}
                title="Không liên quan"
            >
                <ThumbsDown className="w-2.5 h-2.5" />
            </button>
        </div>
    );
}

function SourcesPanel({ sources, messageId }: { sources: ChatSourceChunk[]; messageId?: string }) {
    const [expanded, setExpanded] = useState(false);
    const [ratings, setRatings] = useState<Record<string, RelevanceRating>>({});
    const { activateCitation, activateCitationKG } = useWorkspaceStore();
    const debugMode = useContext(DebugCtx);

    if (sources.length === 0) {
        return null;
    }

    const vectorSources = sources.filter((source) => source.source_type !== "kg");
    const kgSources = sources.filter((source) => source.source_type === "kg");

    const handleRate = (sourceIndex: string, rating: RelevanceRating) => {
        const newRating = ratings[sourceIndex] === rating ? "partial" : rating;
        setRatings((prev) => ({ ...prev, [sourceIndex]: newRating }));

        if (!messageId) {
            return;
        }

        toast.info("Tạm thời chưa đồng bộ được đánh giá nguồn trên server này.");
    };

    return (
        <div className="mt-2 rounded-md border bg-muted/20 overflow-hidden">
            <button onClick={() => setExpanded((prev) => !prev)} className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors">
                <FileText className="w-3 h-3" />
                {vectorSources.length} nguồn
                {kgSources.length > 0 && " + KG"}
                <span className="ml-auto text-[10px]">{expanded ? "▲" : "▼"}</span>
            </button>
            <AnimatePresence>
                {expanded && (
                    <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                        <div className="divide-y border-t">
                            {vectorSources.map((source) => (
                                <button key={source.chunk_id} onClick={() => activateCitation(source, [])} className="w-full text-left px-2.5 py-2 hover:bg-muted/50 transition-colors">
                                    <div className="flex items-center gap-1.5 mb-0.5">
                                        <span className="inline-flex items-center justify-center w-4 h-4 text-[9px] font-bold rounded-full bg-primary/15 text-primary">{source.index}</span>
                                        <span className="text-[10px] text-muted-foreground">p.{source.page_no}</span>
                                        {source.heading_path.length > 0 && <span className="text-[10px] text-muted-foreground/60 truncate">{source.heading_path.join(" > ")}</span>}
                                        {messageId && <SourceRatingButtons sourceIndex={String(source.index)} currentRating={ratings[String(source.index)]} onRate={handleRate} />}
                                    </div>
                                    <p className="text-[11px] text-foreground/70 line-clamp-2 leading-relaxed">
                                        {source.content.slice(0, 150)}
                                        {source.content.length > 150 ? "..." : ""}
                                    </p>
                                    {debugMode && (
                                        <div className="flex items-center gap-1.5 mt-0.5">
                                            <span className="text-[8px] px-1 py-0.5 rounded bg-muted font-mono text-muted-foreground/70">score: {source.score.toFixed(3)}</span>
                                            <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-blue-400/15 text-blue-400">{source.source_type || "vector"}</span>
                                        </div>
                                    )}
                                </button>
                            ))}
                            {kgSources.map((source) => (
                                <button key={source.chunk_id} onClick={() => activateCitationKG(source, [])} className="w-full text-left px-2.5 py-2 hover:bg-purple-400/5 hover:bg-muted/50 transition-colors">
                                    <div className="flex items-center gap-1.5 mb-0.5">
                                        <span className="inline-flex items-center justify-center w-4 h-4 text-[9px] font-bold rounded-full bg-purple-400/15 text-purple-400">{source.index}</span>
                                        <span className="text-[10px] text-purple-400 font-medium">Knowledge Graph</span>
                                        {messageId && <SourceRatingButtons sourceIndex={String(source.index)} currentRating={ratings[String(source.index)]} onRate={handleRate} />}
                                    </div>
                                    <p className="text-[11px] text-foreground/70 line-clamp-2 leading-relaxed">
                                        {source.content.slice(0, 150)}
                                        {source.content.length > 150 ? "..." : ""}
                                    </p>
                                    {debugMode && (
                                        <div className="flex items-center gap-1.5 mt-0.5">
                                            <span className="text-[8px] px-1 py-0.5 rounded bg-muted font-mono text-muted-foreground/70">score: {source.score.toFixed(3)}</span>
                                            <span className="text-[8px] px-1 py-0.5 rounded font-medium bg-purple-400/15 text-purple-400">kg</span>
                                        </div>
                                    )}
                                </button>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function ImageRefCard({ image }: { image: ChatImageRef }) {
    const { activateImageCitation } = useWorkspaceStore();
    const doc = useFindDoc(image.document_id);

    return (
        <button onClick={() => activateImageCitation(image, doc)} className="group block rounded-md overflow-hidden border bg-background hover:border-primary/50 transition-colors text-left cursor-pointer">
            <img src={image.url} alt={image.caption || `Hình từ trang ${image.page_no}`} className="w-full h-auto max-h-[200px] object-contain bg-white" loading="lazy" />
            {image.caption && (
                <p className="px-2 py-1 text-[10px] text-muted-foreground leading-tight line-clamp-2 border-t">
                    p.{image.page_no} - {image.caption}
                </p>
            )}
        </button>
    );
}

function ImageRefsPanel({ images }: { images: ChatImageRef[] }) {
    const [expanded, setExpanded] = useState(true);

    if (images.length === 0) {
        return null;
    }

    return (
        <div className="mt-2 rounded-md border bg-muted/20 overflow-hidden">
            <button onClick={() => setExpanded((prev) => !prev)} className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-muted-foreground hover:text-foreground transition-colors">
                <ImageIcon className="w-3 h-3" />
                {images.length} hình ảnh từ tài liệu
                <span className="ml-auto text-[10px]">{expanded ? "▲" : "▼"}</span>
            </button>
            <AnimatePresence>
                {expanded && (
                    <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                        <div className="p-2 grid gap-2" style={{ gridTemplateColumns: images.length === 1 ? "1fr" : "repeat(auto-fit, minmax(140px, 1fr))" }}>
                            {images.map((image) => (
                                <ImageRefCard key={image.image_id} image={image} />
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

function ThinkingPanel({ thinking }: { thinking: string }) {
    const [expanded, setExpanded] = useState(false);

    if (!thinking) {
        return null;
    }

    return (
        <div className="mt-1.5 mb-1 rounded-md border border-violet-500/20 bg-violet-500/5 overflow-hidden">
            <button
                onClick={() => setExpanded((prev) => !prev)}
                className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-[10px] font-medium text-violet-400 hover:text-violet-300 [[data-theme='light']_&]:text-violet-600 [[data-theme='light']_&]:hover:text-violet-700 transition-colors"
            >
                <Brain className="w-3 h-3" />
                Quá trình suy nghĩ
                <ChevronDown className={cn("w-3 h-3 ml-auto transition-transform", expanded && "rotate-180")} />
            </button>
            <AnimatePresence>
                {expanded && (
                    <motion.div initial={{ height: 0 }} animate={{ height: "auto" }} exit={{ height: 0 }} className="overflow-hidden">
                        <div className="px-2.5 pb-2 border-t border-violet-500/10">
                            <pre className="text-[11px] text-violet-300/90 [[data-theme='light']_&]:text-violet-700/90 whitespace-pre-wrap leading-relaxed mt-1.5 max-h-[300px] overflow-y-auto">{thinking}</pre>
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}

const CITATION_STRIP_RE = /\s*\[(?:[a-z0-9]+|IMG-[a-z0-9]+)(?:,\s*(?:[a-z0-9]+|IMG-[a-z0-9]+))*\]/g;

function stripCitations(markdown: string): string {
    return markdown
        .replace(CITATION_STRIP_RE, "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
}

function markdownToPlainText(markdown: string): string {
    let text = stripCitations(markdown);
    text = text.replace(/```[\s\S]*?```/g, (match) => {
        const lines = match.split("\n");
        return lines.slice(1, -1).join("\n");
    });
    text = text.replace(/`([^`]+)`/g, "$1");
    text = text.replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1");
    text = text.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1");
    text = text.replace(/\*\*(.+?)\*\*/g, "$1");
    text = text.replace(/\*(.+?)\*/g, "$1");
    text = text.replace(/__(.+?)__/g, "$1");
    text = text.replace(/_(.+?)_/g, "$1");
    text = text.replace(/^#{1,6}\s+/gm, "");
    text = text.replace(/^[-*_]{3,}\s*$/gm, "");
    text = text.replace(/\n{3,}/g, "\n\n");
    return text.trim();
}

function CopyMessageActions({ content }: { content: string }) {
    const [copiedMode, setCopiedMode] = useState<"text" | "markdown" | null>(null);

    const handleCopy = useCallback(
        (mode: "text" | "markdown") => {
            const value = mode === "text" ? markdownToPlainText(content) : stripCitations(content);
            navigator.clipboard.writeText(value).then(() => {
                setCopiedMode(mode);
                setTimeout(() => setCopiedMode(null), 2000);
            });
        },
        [content],
    );

    return (
        <div className="flex items-center gap-0.5 mt-1.5">
            <button onClick={() => handleCopy("text")} className="flex items-center gap-1 px-1.5 py-0.5 rounded-md text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/60 transition-all text-[10px]" title="Sao chép dạng văn bản">
                {copiedMode === "text" ? <ClipboardCheck className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
                <span>{copiedMode === "text" ? "Đã sao chép!" : "Sao chép văn bản"}</span>
            </button>
            <button onClick={() => handleCopy("markdown")} className="flex items-center gap-1 px-1.5 py-0.5 rounded-md text-muted-foreground/50 hover:text-muted-foreground hover:bg-muted/60 transition-all text-[10px]" title="Sao chép dạng markdown">
                {copiedMode === "markdown" ? <ClipboardCheck className="w-3 h-3 text-emerald-500" /> : <FileCode className="w-3 h-3" />}
                <span>{copiedMode === "markdown" ? "Đã sao chép!" : "Sao chép markdown"}</span>
            </button>
        </div>
    );
}

function InlineThinkingPreview({ text }: { text: string }) {
    const containerRef = useRef<HTMLDivElement>(null);
    const isUserScrolledRef = useRef(false);

    const handleScroll = useCallback(() => {
        const element = containerRef.current;
        if (!element) {
            return;
        }

        const isAtBottom = element.scrollHeight - element.scrollTop - element.clientHeight < 20;
        isUserScrolledRef.current = !isAtBottom;
    }, []);

    useEffect(() => {
        if (containerRef.current && !isUserScrolledRef.current) {
            containerRef.current.scrollTop = containerRef.current.scrollHeight;
        }
    }, [text]);

    return (
        <div className="mt-1">
            <div className="flex items-center gap-1.5 mb-1.5">
                <Brain className="w-3.5 h-3.5 text-violet-400 animate-pulse" />
                <span className="text-xs font-medium text-violet-400">Đang suy nghĩ...</span>
            </div>
            <div
                ref={containerRef}
                onScroll={handleScroll}
                className={cn("text-xs leading-relaxed text-muted-foreground/70 italic", "max-h-[200px] overflow-y-auto scrollbar-none", "border-l-2 border-violet-500/30 pl-3", "whitespace-pre-wrap break-words")}
            >
                {text}
                <span className="animate-pulse text-violet-400 ml-0.5">|</span>
            </div>
        </div>
    );
}

function TypingIndicator({ status }: { status?: ChatStreamStatus }) {
    const label = (status && STATUS_LABELS[status]) || "Đang phân tích tài liệu...";

    return (
        <div className="flex gap-2 items-start">
            <div className="relative w-6 h-6 flex-shrink-0">
                <div className="icon-glow-ring" />
                <div className="w-6 h-6 rounded-full bg-primary/15 flex items-center justify-center">
                    <Bot className="w-3.5 h-3.5 text-primary" />
                </div>
            </div>
            <div className="py-1">
                <div className="flex items-center gap-1.5">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-primary" />
                    <span className="text-xs text-muted-foreground">{label}</span>
                </div>
            </div>
        </div>
    );
}

export const MessageBubble = memo(function MessageBubble({ message }: { message: ChatMessage }) {
    const isUser = message.role === "user";

    const timelineSteps = useMemo(() => {
        if (isUser) {
            return [];
        }

        if (message.agentSteps?.length) {
            return message.agentSteps;
        }

        return [];
    }, [isUser, message]);

    const proseClasses = cn(
        "prose prose-sm max-w-none text-foreground/90",
        "[&_p]:my-1 [&_ul]:my-1 [&_ol]:my-1 [&_li]:my-0.5",
        "[&_pre]:bg-transparent [&_pre]:border-none [&_pre]:p-0 [&_pre]:m-0",
        "[&_code]:bg-muted/50 [&_code]:px-1 [&_code]:py-0.5 [&_code]:rounded [&_code]:text-xs [&_code]:text-foreground/90",
        "[&_a]:text-primary [&_a]:underline [&_a]:underline-offset-2",
        "[&_strong]:text-foreground [&_em]:text-foreground/80",
        "[&_h1]:text-foreground [&_h2]:text-foreground [&_h3]:text-foreground [&_h4]:text-foreground",
        "[&_h1]:text-base [&_h1]:font-bold [&_h1]:mt-3 [&_h1]:mb-1",
        "[&_h2]:text-sm [&_h2]:font-semibold [&_h2]:mt-2.5 [&_h2]:mb-1",
        "[&_h3]:text-sm [&_h3]:font-semibold [&_h3]:mt-2 [&_h3]:mb-0.5",
        "[&_blockquote]:border-l-2 [&_blockquote]:border-primary/30 [&_blockquote]:pl-3 [&_blockquote]:italic [&_blockquote]:text-foreground/60",
        "[&_table]:text-xs [&_th]:px-2 [&_th]:py-1 [&_td]:px-2 [&_td]:py-1 [&_th]:text-foreground/80 [&_td]:text-foreground/80",
        "[&_li]:text-foreground/90",
        "[&_.katex-display]:overflow-x-auto [&_.katex-display]:py-2",
        "[&_.katex]:text-[0.9em]",
    );

    return (
        <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={cn("flex gap-2", isUser ? "justify-end" : "justify-start")}>
            {!isUser && (
                <div className="relative w-6 h-6 flex-shrink-0 mt-1">
                    {message.isStreaming && <div className="icon-glow-ring" />}
                    <div className="w-6 h-6 rounded-full bg-primary/15 flex items-center justify-center">
                        <Bot className="w-3.5 h-3.5 text-primary" />
                    </div>
                </div>
            )}

            <div className={cn(isUser ? "max-w-[85%] rounded-xl px-3 py-2 bg-secondary/50" : "max-w-[90%] min-w-0 py-1")}>
                {!isUser && timelineSteps.length > 0 && (
                    <ThinkingTimeline steps={timelineSteps} mode={message.isStreaming ? "live" : "embedded"} className={cn("mb-1.5", message.isStreaming && "mt-1")} autoCollapse={message.isStreaming && !!message.content} />
                )}

                {!isUser && message.isStreaming && !message.content && !message.agentSteps?.length && <TypingIndicator status="analyzing" />}

                {isUser ? (
                    <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                ) : message.isStreaming ? (
                    message.content ? (
                        <div
                            className={cn(proseClasses, "relative")}
                            style={{
                                maskImage: "linear-gradient(to bottom, black calc(100% - 80px), transparent 100%)",
                                WebkitMaskImage: "linear-gradient(to bottom, black calc(100% - 80px), transparent 100%)",
                            }}
                        >
                            <StreamingMarkdown
                                content={message.content}
                                isStreaming
                                renderBlock={(block) => <MarkdownWithCitations content={block} sources={message.sources || []} relatedEntities={message.relatedEntities || []} imageRefs={message.imageRefs} />}
                            />
                            <span className="streaming-cursor" />
                        </div>
                    ) : message.thinking ? (
                        <InlineThinkingPreview text={message.thinking} />
                    ) : null
                ) : (
                    <div className={proseClasses}>
                        <MarkdownWithCitations content={message.content} sources={message.sources || []} relatedEntities={message.relatedEntities || []} imageRefs={message.imageRefs} />
                    </div>
                )}

                {!isUser && message.content && <CopyMessageActions content={message.content} />}
                {!isUser && message.thinking && !message.isStreaming && !message.agentSteps?.some((step) => step.thinkingText) && <ThinkingPanel thinking={message.thinking} />}
                {!isUser && !message.isStreaming && message.sources && message.sources.length > 0 && <SourcesPanel sources={message.sources} messageId={message.id} />}
                {!isUser && !message.isStreaming && message.imageRefs && message.imageRefs.length > 0 && <ImageRefsPanel images={message.imageRefs} />}

                <p className={cn("text-[9px] mt-1", isUser ? "text-muted-foreground/50" : "text-muted-foreground/50")}>
                    {new Date(message.timestamp).toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                    })}
                </p>
            </div>

            {isUser && (
                <div className="w-6 h-6 rounded-full bg-secondary flex items-center justify-center flex-shrink-0 mt-1">
                    <User className="w-3.5 h-3.5 text-muted-foreground" />
                </div>
            )}
        </motion.div>
    );
});

export function SuggestionChips({ onSelect }: { onSelect: (question: string) => void }) {
    const suggestions = ["Tóm tắt các phát hiện chính", "Chủ đề chính trong tài liệu là gì?", "Liệt kê các entity quan trọng", "Giải thích phương pháp được sử dụng"];

    return (
        <div className="flex-1 flex flex-col items-center justify-center px-4">
            <div className="w-12 h-12 rounded-2xl bg-primary/10 flex items-center justify-center mb-4">
                <Sparkles className="w-6 h-6 text-primary" />
            </div>
            <h3 className="text-sm font-semibold mb-1">Trợ lý tài liệu AI</h3>
            <p className="text-xs text-muted-foreground text-center mb-4 max-w-[240px]">Đặt câu hỏi về tài liệu của bạn. Tôi sẽ tìm thông tin liên quan và trích dẫn nguồn.</p>
            <div className="flex flex-wrap gap-1.5 justify-center max-w-[300px]">
                {suggestions.map((suggestion) => (
                    <button key={suggestion} onClick={() => onSelect(suggestion)} className="text-[11px] px-2.5 py-1 rounded-full border bg-card hover:bg-muted transition-colors text-muted-foreground hover:text-foreground">
                        {suggestion}
                    </button>
                ))}
            </div>
        </div>
    );
}
