import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { ChatHistoryResponse, ChatSourceChunk, PersistedChatMessage } from "@/types";

interface ServerCitationItem {
    document_id: string;
    filename: string;
    chunk_text: string;
    relevance_score: number;
}

interface ServerChatMessage {
    id: string;
    role: "user" | "assistant";
    content: string;
    citations?: ServerCitationItem[] | null;
    created_at: string;
}

interface ServerSessionDetail {
    id: string;
    messages: ServerChatMessage[];
}

function mapCitationToSource(citation: ServerCitationItem, index: number): ChatSourceChunk {
    return {
        index: String(index + 1),
        chunk_id: `${citation.document_id}-${index + 1}`,
        content: citation.chunk_text || citation.filename,
        document_id: citation.document_id,
        page_no: null,
        heading_path: [],
        score: citation.relevance_score ?? 0,
        source_type: "vector",
    };
}

function mapMessage(message: ServerChatMessage): PersistedChatMessage {
    const sources = message.citations?.map(mapCitationToSource) ?? null;

    return {
        id: message.id,
        message_id: message.id,
        role: message.role,
        content: message.content,
        sources,
        related_entities: null,
        image_refs: null,
        thinking: null,
        agent_steps: null,
        created_at: message.created_at,
    };
}

export function useChatHistory(workspaceId: string) {
    return useQuery({
        queryKey: ["chat-history", workspaceId],
        queryFn: async () => {
            const session = await api.get<ServerSessionDetail>(`/chat/sessions/${workspaceId}`);
            const messages = session.messages.map(mapMessage);
            return {
                workspace_id: session.id,
                messages,
                total: messages.length,
            } as ChatHistoryResponse;
        },
        enabled: !!workspaceId,
        staleTime: Infinity, // Don't auto-refetch — we invalidate manually after chat
    });
}

export function useClearChatHistory(workspaceId: string) {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async () => api.delete(`/chat/sessions/${workspaceId}/messages`),
        onSuccess: () => {
            queryClient.setQueryData<ChatHistoryResponse>(["chat-history", workspaceId], {
                workspace_id: workspaceId,
                messages: [],
                total: 0,
            });
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            queryClient.invalidateQueries({ queryKey: ["workspaces", "summary"] });
        },
    });
}
