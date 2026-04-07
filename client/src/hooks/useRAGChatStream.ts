import { useState, useRef, useCallback, useEffect } from "react";
import { generateId } from "@/lib/utils";
import type { ChatSourceChunk, ChatImageRef, ChatStreamStatus, ChatMessage, AgentStep, AgentStepType } from "@/types";

const BASE_URL = import.meta.env.VITE_API_URL || "/api/v1";

interface ServerCitationItem {
    document_id: string;
    filename: string;
    chunk_text: string;
    relevance_score: number;
}

interface ServerStreamEvent {
    type: "message_ids" | "token" | "citations" | "done" | "error";
    data?: unknown;
}

export interface RAGStreamResult {
    status: ChatStreamStatus;
    streamingContent: string;
    thinkingText: string;
    pendingSources: ChatSourceChunk[];
    pendingImages: ChatImageRef[];
    error: string | null;
    isStreaming: boolean;
    agentSteps: AgentStep[];
    sendMessage: (message: string, history: { role: string; content: string }[], enableThinking: boolean, forceSearch?: boolean) => Promise<ChatMessage | null>;
    cancel: () => void;
    reset: () => void;
}

function createStep(step: AgentStepType, detail: string, status: "active" | "completed" | "error" = "active"): AgentStep {
    return {
        id: generateId(),
        step,
        detail,
        status,
        timestamp: Date.now(),
    };
}

function completeActiveStep(steps: AgentStep[]): AgentStep[] {
    const now = Date.now();
    return steps.map((s) => (s.status === "active" ? { ...s, status: "completed" as const, durationMs: now - s.timestamp } : s));
}

function markActiveError(steps: AgentStep[]): AgentStep[] {
    return steps.map((s) => (s.status === "active" ? { ...s, status: "error" as const } : s));
}

function mapCitationsToSources(citations: ServerCitationItem[]): ChatSourceChunk[] {
    return citations.map((citation, index) => ({
        index: String(index + 1),
        chunk_id: `${citation.document_id}-${index + 1}`,
        content: citation.chunk_text || citation.filename,
        document_id: citation.document_id,
        page_no: null,
        heading_path: [],
        score: citation.relevance_score ?? 0,
        source_type: "vector",
    }));
}

export function useRAGChatStream(workspaceId: string): RAGStreamResult {
    const [status, setStatus] = useState<ChatStreamStatus>("idle");
    const [streamingContent, setStreamingContent] = useState("");
    const [thinkingText, setThinkingText] = useState("");
    const [pendingSources, setPendingSources] = useState<ChatSourceChunk[]>([]);
    const [pendingImages, setPendingImages] = useState<ChatImageRef[]>([]);
    const [error, setError] = useState<string | null>(null);
    const [isStreaming, setIsStreaming] = useState(false);
    const [agentSteps, setAgentSteps] = useState<AgentStep[]>([]);

    const abortRef = useRef<AbortController | null>(null);
    const bufferRef = useRef("");
    const rafRef = useRef<number | undefined>(undefined);
    const streamStartRef = useRef(0);

    useEffect(() => {
        return () => {
            abortRef.current?.abort();
            if (rafRef.current) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    const syncSteps = useCallback((next: AgentStep[] | ((prev: AgentStep[]) => AgentStep[])) => {
        setAgentSteps((prev) => (typeof next === "function" ? next(prev) : next));
    }, []);

    const flushTokenBuffer = useCallback(() => {
        if (!bufferRef.current) return;
        const remaining = bufferRef.current;
        bufferRef.current = "";
        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = undefined;
        }
        setStreamingContent((prev) => prev + remaining);
    }, []);

    const onToken = useCallback((text: string) => {
        bufferRef.current += text;
        if (!rafRef.current) {
            rafRef.current = requestAnimationFrame(() => {
                const chunk = bufferRef.current;
                bufferRef.current = "";
                rafRef.current = undefined;
                setStreamingContent((prev) => prev + chunk);
            });
        }
    }, []);

    const reset = useCallback(() => {
        setStatus("idle");
        setStreamingContent("");
        setThinkingText("");
        setPendingSources([]);
        setPendingImages([]);
        setError(null);
        setIsStreaming(false);
        setAgentSteps([]);
        bufferRef.current = "";
        if (rafRef.current) {
            cancelAnimationFrame(rafRef.current);
            rafRef.current = undefined;
        }
    }, []);

    const cancel = useCallback(() => {
        abortRef.current?.abort();
        abortRef.current = null;
        flushTokenBuffer();
        setStatus("idle");
        setIsStreaming(false);
    }, [flushTokenBuffer]);

    const sendMessage = useCallback(
        async (message: string, _history: { role: string; content: string }[], _enableThinking: boolean, _forceSearch: boolean = false): Promise<ChatMessage | null> => {
            setStreamingContent("");
            setThinkingText("");
            setPendingSources([]);
            setPendingImages([]);
            setError(null);
            setStatus("analyzing");
            setIsStreaming(true);
            setAgentSteps([]);
            bufferRef.current = "";
            streamStartRef.current = Date.now();

            let localSteps: AgentStep[] = [createStep("analyzing", "Preparing retrieval and generation...")];
            syncSteps(localSteps);

            let finalSources: ChatSourceChunk[] = [];
            let fullAnswer = "";
            let hasGenerationStep = false;
            let assistantMessageIdFromServer: string | null = null;

            abortRef.current = new AbortController();

            try {
                const response = await fetch(`${BASE_URL}/chat/stream`, {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        session_id: workspaceId,
                        query: message,
                        stream: true,
                    }),
                    signal: abortRef.current.signal,
                });

                if (!response.ok) {
                    const err = await response.json().catch(() => ({ detail: "Stream request failed" }));
                    throw new Error(err.detail || `Error: ${response.status}`);
                }

                const reader = response.body?.getReader();
                if (!reader) throw new Error("No response body");

                const decoder = new TextDecoder();
                let sseBuffer = "";
                let doneEventReceived = false;

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    sseBuffer += decoder.decode(value, { stream: true });
                    const lines = sseBuffer.split("\n");
                    sseBuffer = lines.pop() || "";

                    for (const line of lines) {
                        if (!line.startsWith("data: ")) continue;

                        const jsonStr = line.slice(6).trim();
                        if (!jsonStr) continue;

                        let event: ServerStreamEvent;
                        try {
                            event = JSON.parse(jsonStr) as ServerStreamEvent;
                        } catch {
                            continue;
                        }

                        if (event.type === "message_ids") {
                            const data = event.data as { assistant_message_id?: unknown } | undefined;
                            if (data && typeof data.assistant_message_id === "string") {
                                assistantMessageIdFromServer = data.assistant_message_id;
                            }
                            if (!hasGenerationStep) {
                                localSteps = [...completeActiveStep(localSteps), createStep("retrieving", "Retrieved supporting context", "completed")];
                                syncSteps(localSteps);
                            }
                            continue;
                        }

                        if (event.type === "token") {
                            const token = String(event.data ?? "");
                            if (!hasGenerationStep) {
                                setStatus("generating");
                                hasGenerationStep = true;
                                localSteps = [...completeActiveStep(localSteps), createStep("generating", "Generating answer...")];
                                syncSteps(localSteps);
                            }
                            fullAnswer += token;
                            onToken(token);
                            continue;
                        }

                        if (event.type === "citations") {
                            const citations = Array.isArray(event.data) ? (event.data as ServerCitationItem[]) : [];
                            finalSources = mapCitationsToSources(citations);
                            setPendingSources(finalSources);

                            localSteps = [
                                ...completeActiveStep(localSteps),
                                {
                                    ...createStep("sources_found", `Found ${finalSources.length} source${finalSources.length === 1 ? "" : "s"}`, "completed"),
                                    sourceBadges: finalSources.map((s) => String(s.index)),
                                    sourceCount: finalSources.length,
                                },
                                createStep("generating", "Finalizing answer..."),
                            ];
                            syncSteps(localSteps);
                            continue;
                        }

                        if (event.type === "error") {
                            throw new Error(String(event.data ?? "Unknown stream error"));
                        }

                        if (event.type === "done") {
                            doneEventReceived = true;
                        }
                    }

                    if (doneEventReceived) break;
                }

                flushTokenBuffer();

                const totalMs = Date.now() - streamStartRef.current;
                localSteps = [...completeActiveStep(localSteps), createStep("done", `Done in ${totalMs >= 1000 ? `${(totalMs / 1000).toFixed(1)}s` : `${totalMs}ms`}`, "completed")];
                syncSteps(localSteps);

                setStatus("idle");
                setIsStreaming(false);

                return {
                    id: assistantMessageIdFromServer ?? generateId(),
                    role: "assistant",
                    content: fullAnswer,
                    sources: finalSources,
                    relatedEntities: [],
                    imageRefs: [],
                    thinking: null,
                    agentSteps: localSteps,
                    timestamp: new Date().toISOString(),
                };
            } catch (err) {
                if ((err as Error).name === "AbortError") {
                    return null;
                }

                const msg = (err as Error).message || "Stream failed";
                setError(msg);
                setStatus("error");
                setIsStreaming(false);
                syncSteps((prev) => markActiveError(prev));
                return null;
            } finally {
                abortRef.current = null;
            }
        },
        [workspaceId, flushTokenBuffer, onToken, syncSteps],
    );

    return {
        status,
        streamingContent,
        thinkingText,
        pendingSources,
        pendingImages,
        error,
        isStreaming,
        agentSteps,
        sendMessage,
        cancel,
        reset,
    };
}
