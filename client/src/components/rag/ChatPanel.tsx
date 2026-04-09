import { useState, useRef, useEffect, useCallback, useMemo, memo } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Send, Square, Bot, Loader2, Trash2, Brain, Settings, RotateCcw, Info, Save, DatabaseZap } from "lucide-react";
import { toast } from "sonner";
import { cn, generateId } from "@/lib/utils";
import { useWorkspaceStore } from "@/stores/workspaceStore";
import { useChatHistory, useClearChatHistory } from "@/hooks/useChatHistory";
import { useRAGChatStream } from "@/hooks/useRAGChatStream";
import type { ChatMessage, ChatSourceChunk, KnowledgeBase, AgentStep } from "@/types";
import { AllSourcesCtx, DebugCtx, WsIdCtx } from "@/components/rag/chat-panel/context";
import { DEFAULT_SYSTEM_PROMPT, HARD_RULES_SUMMARY } from "@/components/rag/chat-panel/constants";
import { MessageBubble, SuggestionChips } from "@/components/rag/chat-panel/messageBlocks";
import { ConfirmDialog } from "@/components/ui/confirm-dialog";

interface ChatPanelProps {
    workspaceId: string;
    hasIndexedDocs: boolean;
    workspace: KnowledgeBase | null;
}

export const ChatPanel = memo(function ChatPanel({ workspaceId, hasIndexedDocs, workspace }: ChatPanelProps) {
    const [messages, setMessages] = useState<ChatMessage[]>([]);
    const [input, setInput] = useState("");
    const [enableThinking, setEnableThinking] = useState(false);
    const [forceSearch, setForceSearch] = useState(false);

    const { data: historyData, isLoading: historyLoading } = useChatHistory(workspaceId);
    const clearMutation = useClearChatHistory(workspaceId);
    const [showPromptEditor, setShowPromptEditor] = useState(false);
    const [clearChatConfirmOpen, setClearChatConfirmOpen] = useState(false);
    const [promptDraft, setPromptDraft] = useState("");
    const scrollContainerRef = useRef<HTMLDivElement>(null);
    const scrollAnimRef = useRef<number | undefined>(undefined);
    const spacerRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLTextAreaElement>(null);

    const [debugMode, setDebugMode] = useState(() => localStorage.getItem("nexusrag-debug-mode") === "true");

    useEffect(() => {
        const handler = (event: KeyboardEvent) => {
            if (event.ctrlKey && event.shiftKey && event.key === "D") {
                event.preventDefault();
                setDebugMode((prev) => {
                    const next = !prev;
                    localStorage.setItem("nexusrag-debug-mode", String(next));
                    toast.success(next ? "Đã bật Debug mode" : "Đã tắt Debug mode");
                    return next;
                });
            }
        };

        window.addEventListener("keydown", handler);
        return () => window.removeEventListener("keydown", handler);
    }, []);

    const savedPrompt = workspace?.system_prompt ?? "";
    const effectivePrompt = savedPrompt || DEFAULT_SYSTEM_PROMPT;
    const isCustom = !!savedPrompt;

    useEffect(() => {
        setPromptDraft(effectivePrompt);
    }, [effectivePrompt]);

    const promptIsDirty = promptDraft !== effectivePrompt;

    const handleSavePrompt = useCallback(() => {
        toast.info("Server hiện chưa hỗ trợ lưu system prompt.");
    }, []);

    const handleResetPrompt = useCallback(() => {
        setPromptDraft(DEFAULT_SYSTEM_PROMPT);
        toast.info("Đã đặt lại prompt trong editor. Bấm Lưu để ghi nhận.");
    }, []);

    const thinkingSupported = false;

    useEffect(() => {
        if (historyData?.messages) {
            setMessages((prev) => {
                const stepsMap = new Map<string, AgentStep[]>();
                for (const message of prev) {
                    if (message.agentSteps?.length) {
                        stepsMap.set(message.id, message.agentSteps);
                    }
                }

                return historyData.messages.map((message) => ({
                    id: message.message_id,
                    role: message.role as "user" | "assistant",
                    content: message.content,
                    sources: message.sources ?? undefined,
                    relatedEntities: message.related_entities ?? undefined,
                    imageRefs: message.image_refs ?? undefined,
                    thinking: message.thinking ?? undefined,
                    timestamp: message.created_at,
                    agentSteps: stepsMap.get(message.message_id) ?? (message.agent_steps?.length ? (message.agent_steps as AgentStep[]) : undefined),
                }));
            });
        }
    }, [historyData]);

    const stream = useRAGChatStream(workspaceId);
    const streamingMsgIdRef = useRef<string | null>(null);
    const agentStepsRef = useRef<AgentStep[]>([]);

    useEffect(() => {
        if (stream.agentSteps.length > 0) {
            agentStepsRef.current = stream.agentSteps;
        }
    }, [stream.agentSteps]);

    const scrollToBottom = useCallback((smooth = true) => {
        const container = scrollContainerRef.current;
        if (!container) {
            return;
        }

        if (scrollAnimRef.current) {
            cancelAnimationFrame(scrollAnimRef.current);
            scrollAnimRef.current = undefined;
        }

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const element = scrollContainerRef.current;
                if (!element) {
                    return;
                }

                const target = element.scrollHeight - element.clientHeight;
                if (!smooth || Math.abs(target - element.scrollTop) < 10) {
                    element.scrollTop = target;
                    return;
                }

                const start = element.scrollTop;
                const distance = target - start;
                const duration = 400;
                const startTime = performance.now();

                const scrollElement = element;
                function animate(now: number) {
                    const t = Math.min((now - startTime) / duration, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    scrollElement.scrollTop = start + distance * ease;
                    if (t < 1) {
                        scrollAnimRef.current = requestAnimationFrame(animate);
                    } else {
                        scrollAnimRef.current = undefined;
                    }
                }

                scrollAnimRef.current = requestAnimationFrame(animate);
            });
        });
    }, []);

    const scrollUserMsgToTop = useCallback((msgId: string) => {
        if (scrollAnimRef.current) {
            cancelAnimationFrame(scrollAnimRef.current);
            scrollAnimRef.current = undefined;
        }

        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                const container = scrollContainerRef.current;
                if (!container) {
                    return;
                }

                if (spacerRef.current) {
                    spacerRef.current.style.height = `${container.clientHeight}px`;
                }

                const messageElement = container.querySelector(`[data-message-id="${msgId}"]`) as HTMLElement | null;
                if (!messageElement) {
                    return;
                }

                const containerRect = container.getBoundingClientRect();
                const messageRect = messageElement.getBoundingClientRect();
                const relativeTop = messageRect.top - containerRect.top + container.scrollTop;

                const start = container.scrollTop;
                const target = Math.max(0, relativeTop - 12);
                if (Math.abs(target - start) < 5) {
                    return;
                }

                const distance = target - start;
                const duration = 380;
                const startTime = performance.now();
                const scrollContainer = container;

                function animate(now: number) {
                    const t = Math.min((now - startTime) / duration, 1);
                    const ease = 1 - Math.pow(1 - t, 3);
                    scrollContainer.scrollTop = start + distance * ease;
                    if (t < 1) {
                        scrollAnimRef.current = requestAnimationFrame(animate);
                    } else {
                        scrollAnimRef.current = undefined;
                    }
                }

                scrollAnimRef.current = requestAnimationFrame(animate);
            });
        });
    }, []);

    const hasMessages = messages.length > 0;
    useEffect(() => {
        if (!hasMessages) {
            return;
        }

        const container = scrollContainerRef.current;
        const spacer = spacerRef.current;
        if (!container || !spacer) {
            return;
        }

        if (!stream.isStreaming) {
            spacer.style.height = "0px";
            return;
        }

        const updateSpacer = () => {
            spacer.style.height = `${container.clientHeight}px`;
        };

        updateSpacer();
        const observer = new ResizeObserver(updateSpacer);
        observer.observe(container);
        return () => observer.disconnect();
    }, [hasMessages, stream.isStreaming]);

    const prevIsStreamingRef = useRef(false);
    const justFinishedStreamingRef = useRef(false);
    useEffect(() => {
        if (prevIsStreamingRef.current && !stream.isStreaming) {
            if (spacerRef.current) {
                spacerRef.current.style.height = "0px";
            }
            justFinishedStreamingRef.current = true;
        }

        prevIsStreamingRef.current = stream.isStreaming;
    }, [stream.isStreaming]);

    useEffect(() => {
        if (!stream.isStreaming) {
            if (justFinishedStreamingRef.current) {
                justFinishedStreamingRef.current = false;
                return;
            }
            scrollToBottom();
        }
    }, [messages, stream.isStreaming, scrollToBottom]);

    useEffect(() => {
        if (!stream.isStreaming || !streamingMsgIdRef.current) {
            return;
        }

        const messageId = streamingMsgIdRef.current;
        setMessages((prev) => {
            const idx = prev.findIndex((message) => message.id === messageId);
            if (idx === -1) {
                return prev;
            }

            const current = prev[idx];
            const newContent = stream.streamingContent;
            const newSources = stream.pendingSources.length > 0 ? stream.pendingSources : current.sources;
            const newImages = stream.pendingImages.length > 0 ? stream.pendingImages : current.imageRefs;
            const newThinking = stream.thinkingText || current.thinking;
            const newSteps = stream.agentSteps.length > 0 ? stream.agentSteps : current.agentSteps;

            if (current.content === newContent && current.sources === newSources && current.imageRefs === newImages && current.thinking === newThinking && current.agentSteps === newSteps) {
                return prev;
            }

            const updated = [...prev];
            updated[idx] = {
                ...current,
                content: newContent,
                sources: newSources,
                imageRefs: newImages,
                thinking: newThinking,
                agentSteps: newSteps,
            };

            return updated;
        });
    }, [stream.streamingContent, stream.pendingSources, stream.pendingImages, stream.thinkingText, stream.isStreaming, stream.agentSteps]);

    const handleSend = useCallback(
        async (text?: string) => {
            const messageText = (text || input).trim();
            if (!messageText || stream.isStreaming) {
                return;
            }

            const userMessage: ChatMessage = {
                id: generateId(),
                role: "user",
                content: messageText,
                timestamp: new Date().toISOString(),
            };

            const assistantId = generateId();
            streamingMsgIdRef.current = assistantId;
            const placeholderMessage: ChatMessage = {
                id: assistantId,
                role: "assistant",
                content: "",
                timestamp: new Date().toISOString(),
                isStreaming: true,
            };

            setMessages((prev) => [...prev, userMessage, placeholderMessage]);
            setInput("");
            scrollUserMsgToTop(userMessage.id);

            const history = messages.map((message) => ({
                role: message.role,
                content: message.content,
            }));

            const finalMessage = await stream.sendMessage(messageText, history, thinkingSupported && enableThinking, forceSearch);

            if (finalMessage) {
                setMessages((prev) =>
                    prev.map((message) =>
                        message.id === assistantId
                            ? {
                                  ...finalMessage,
                                  id: finalMessage.id || assistantId,
                                  isStreaming: false,
                                  agentSteps: finalMessage.agentSteps?.length ? finalMessage.agentSteps : agentStepsRef.current.length > 0 ? agentStepsRef.current : message.agentSteps,
                              }
                            : message,
                    ),
                );
            } else if (stream.error) {
                toast.error("Chat thất bại: " + stream.error);
                setMessages((prev) =>
                    prev.map((message) =>
                        message.id === assistantId
                            ? {
                                  ...message,
                                  content: message.content || "Xin loi, da xay ra loi. Vui long thu lai.",
                                  isStreaming: false,
                              }
                            : message,
                    ),
                );
            } else {
                setMessages((prev) => prev.map((message) => (message.id === assistantId ? { ...message, isStreaming: false } : message)));
            }

            streamingMsgIdRef.current = null;
        },
        [input, messages, stream, thinkingSupported, enableThinking, forceSearch, scrollUserMsgToTop],
    );

    const handleKeyDown = (event: React.KeyboardEvent) => {
        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            handleSend();
        }
    };

    const handleClear = () => {
        setMessages([]);
        clearMutation.mutate();
        useWorkspaceStore.getState().clearHighlights();
        setClearChatConfirmOpen(false);
    };

    const allSources = useMemo(() => {
        const seen = new Set<string>();
        const merged: ChatSourceChunk[] = [];

        for (const message of messages) {
            if (message.role === "assistant" && message.sources) {
                for (const source of message.sources) {
                    const key = String(source.index);
                    if (!seen.has(key)) {
                        seen.add(key);
                        merged.push(source);
                    }
                }
            }
        }

        return merged;
    }, [messages]);

    if (historyLoading) {
        return (
            <div className="h-full flex items-center justify-center border-r">
                <Loader2 className="w-5 h-5 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (!hasIndexedDocs) {
        return (
            <div className="h-full flex flex-col items-center justify-center px-4 border-r">
                <Bot className="w-10 h-10 text-muted-foreground/30 mb-3" />
                <p className="text-sm text-muted-foreground text-center">Hãy index tài liệu để bắt đầu chat</p>
                <p className="text-[11px] text-muted-foreground/60 mt-1">Upload và xử lý tài liệu trong cột dữ liệu</p>
            </div>
        );
    }

    return (
        <WsIdCtx.Provider value={workspaceId}>
            <DebugCtx.Provider value={debugMode}>
                <AllSourcesCtx.Provider value={allSources}>
                    <div className="h-full flex flex-col border-r min-h-0">
                        <div className="flex-shrink-0 flex items-center justify-between px-3 py-2 border-b">
                            <div className="flex items-center gap-2">
                                <Bot className="w-4 h-4 text-primary" />
                                <span className="text-sm font-semibold">Trợ lý AI</span>
                            </div>
                            <div className="flex items-center gap-1.5">
                                {thinkingSupported && (
                                    <button
                                        onClick={() => setEnableThinking((prev) => !prev)}
                                        className={cn("flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors", enableThinking ? "text-violet-400 bg-violet-400/10 hover:bg-violet-400/15" : "text-muted-foreground hover:bg-muted")}
                                        title={enableThinking ? "Đã bật chế độ Thinking" : "Đã tắt chế độ Thinking"}
                                    >
                                        <Brain className="w-3 h-3" />
                                        <span>Thinking</span>
                                    </button>
                                )}
                                <button
                                    onClick={() => setForceSearch((prev) => !prev)}
                                    className={cn("flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors", forceSearch ? "text-amber-500 bg-amber-500/10 hover:bg-amber-500/15" : "text-muted-foreground hover:bg-muted")}
                                    title={forceSearch ? "Đã bật Force Search — tìm kiếm trước mỗi câu trả lời" : "Đã tắt Force Search — AI tự quyết định khi nào tìm kiếm"}
                                >
                                    <DatabaseZap className="w-3 h-3" />
                                    <span>Tìm kiếm</span>
                                </button>
                                <button
                                    onClick={() => setShowPromptEditor((prev) => !prev)}
                                    className={cn("flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] transition-colors", showPromptEditor ? "text-blue-500 bg-blue-500/10 hover:bg-blue-500/15" : "text-muted-foreground hover:bg-muted")}
                                    title="Cai dat system prompt"
                                >
                                    <Settings className="w-3 h-3" />
                                </button>
                                {messages.length > 0 && (
                                    <button onClick={() => setClearChatConfirmOpen(true)} className="p-1 rounded hover:bg-muted transition-colors" title="Xóa chat">
                                        <Trash2 className="w-3.5 h-3.5 text-muted-foreground" />
                                    </button>
                                )}
                                {debugMode && <span className="text-[8px] px-1 py-0.5 rounded bg-amber-500/15 text-amber-500 font-mono font-semibold">DEBUG</span>}
                            </div>
                        </div>

                        <AnimatePresence>
                            {showPromptEditor && (
                                <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="flex-shrink-0 overflow-visible border-b relative z-10">
                                    <div className="px-3 py-2 space-y-2 bg-muted/20">
                                        <div className="flex items-center justify-between">
                                            <span className="text-[11px] font-medium text-muted-foreground">System Prompt</span>
                                            <span className={cn("text-[9px] px-1.5 py-0.5 rounded-full font-medium", isCustom ? "bg-blue-500/15 text-blue-600 dark:text-blue-400" : "bg-muted text-muted-foreground/50")}>
                                                {isCustom ? "Tuy chinh" : "Mac dinh"}
                                            </span>
                                        </div>
                                        <textarea
                                            value={promptDraft}
                                            onChange={(event) => setPromptDraft(event.target.value)}
                                            placeholder="Nhap system prompt tuy chinh..."
                                            rows={8}
                                            className={cn(
                                                "w-full resize-none rounded-md border border-input bg-background px-2.5 py-2 text-xs",
                                                "placeholder:text-muted-foreground/40 focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                                                "leading-relaxed",
                                            )}
                                        />
                                        <div className="flex items-center gap-1.5">
                                            <div className="relative group/cite">
                                                <div className="flex items-center gap-1 cursor-help">
                                                    <Info className="w-3.5 h-3.5 text-blue-600 dark:text-blue-400" />
                                                    <span className="text-[10px] text-blue-600 dark:text-blue-400 font-medium">Hard rules tu dong duoc them</span>
                                                </div>
                                                <div className="absolute left-0 top-full mt-1.5 z-50 w-[340px] rounded-lg border border-border bg-background shadow-xl opacity-0 pointer-events-none group-hover/cite:opacity-100 group-hover/cite:pointer-events-auto transition-opacity duration-150">
                                                    <div className="px-3 py-2.5">
                                                        <p className="text-[10px] font-semibold text-blue-700 dark:text-blue-300 mb-1.5">Citation + Dinh dang + Rang buoc (luon ap dung)</p>
                                                        <ul className="space-y-1">
                                                            {HARD_RULES_SUMMARY.map((rule, index) => (
                                                                <li key={index} className="text-[10px] text-foreground/70 leading-snug flex gap-1">
                                                                    <span className="text-blue-500 dark:text-blue-400 flex-shrink-0">•</span>
                                                                    {rule}
                                                                </li>
                                                            ))}
                                                        </ul>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                        <div className="flex items-center gap-1.5 justify-end">
                                            <button
                                                onClick={handleResetPrompt}
                                                disabled={!isCustom && !promptIsDirty}
                                                className={cn(
                                                    "flex items-center gap-1 px-2 py-1 rounded text-[10px] transition-colors",
                                                    isCustom || promptIsDirty ? "text-muted-foreground hover:bg-muted hover:text-foreground" : "text-muted-foreground/30 cursor-not-allowed",
                                                )}
                                                title="Dat lai prompt mac dinh"
                                            >
                                                <RotateCcw className="w-3 h-3" />
                                                Dat lai
                                            </button>
                                            <button
                                                onClick={handleSavePrompt}
                                                disabled={!promptIsDirty}
                                                className={cn(
                                                    "flex items-center gap-1 px-2.5 py-1 rounded text-[10px] font-medium transition-colors",
                                                    promptIsDirty ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-muted text-muted-foreground/50 cursor-not-allowed",
                                                )}
                                            >
                                                <Save className="w-3 h-3" />
                                                Luu
                                            </button>
                                        </div>
                                    </div>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {messages.length === 0 ? (
                            <SuggestionChips onSelect={handleSend} />
                        ) : (
                            <div ref={scrollContainerRef} className="flex-1 min-h-0 overflow-y-auto overscroll-contain px-3 py-3 space-y-3 relative bg-background">
                                <AnimatePresence>
                                    {messages.map((message) => (
                                        <div key={message.id} data-message-id={message.id}>
                                            <MessageBubble message={message} />
                                        </div>
                                    ))}
                                </AnimatePresence>
                                <div ref={spacerRef} aria-hidden />
                            </div>
                        )}

                        <div className="flex-shrink-0 p-3 border-t">
                            <div className="flex items-end gap-2">
                                <textarea
                                    ref={inputRef}
                                    value={input}
                                    onChange={(event) => setInput(event.target.value)}
                                    onKeyDown={handleKeyDown}
                                    placeholder="Đặt câu hỏi về tài liệu của bạn..."
                                    rows={1}
                                    className={cn(
                                        "flex-1 resize-none rounded-lg border border-input bg-background px-3 py-2 text-sm",
                                        "placeholder:text-muted-foreground focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring",
                                        "max-h-[120px] min-h-[36px]",
                                    )}
                                    style={{
                                        height: "auto",
                                        minHeight: "36px",
                                    }}
                                    onInput={(event) => {
                                        const target = event.target as HTMLTextAreaElement;
                                        target.style.height = "auto";
                                        target.style.height = Math.min(target.scrollHeight, 120) + "px";
                                    }}
                                />
                                {stream.isStreaming ? (
                                    <button onClick={stream.cancel} className="flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center transition-colors bg-destructive/15 text-destructive hover:bg-destructive/25" title="Dung tao phan hoi">
                                        <Square className="w-3.5 h-3.5 fill-current" />
                                    </button>
                                ) : (
                                    <button
                                        onClick={() => handleSend()}
                                        disabled={!input.trim()}
                                        className={cn(
                                            "flex-shrink-0 w-9 h-9 rounded-lg flex items-center justify-center transition-colors",
                                            input.trim() ? "bg-primary text-primary-foreground hover:bg-primary/90" : "bg-muted text-muted-foreground cursor-not-allowed",
                                        )}
                                    >
                                        <Send className="w-4 h-4" />
                                    </button>
                                )}
                            </div>
                            <p className="text-[9px] text-muted-foreground/50 mt-1 text-center">Nhan Enter de gui, Shift+Enter de xuong dong</p>
                        </div>
                    </div>
                </AllSourcesCtx.Provider>
            </DebugCtx.Provider>

            <ConfirmDialog
                open={clearChatConfirmOpen}
                onConfirm={handleClear}
                onCancel={() => setClearChatConfirmOpen(false)}
                title="Xóa lịch sử chat"
                message="Bạn có chắc chắn muốn xóa toàn bộ lịch sử chat trong workspace này không?"
                confirmLabel="Xóa"
                cancelLabel="Hủy"
                variant="danger"
            />
        </WsIdCtx.Provider>
    );
});
