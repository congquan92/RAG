import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/lib/api";
import type { KnowledgeBase, CreateWorkspace, UpdateWorkspace, WorkspaceSummary } from "@/types";

interface ServerChatSession {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    message_count: number;
    description?: string | null;
    system_prompt?: string | null;
    kg_language?: string | null;
    kg_entity_types?: string[] | null;
}

interface ServerChatSessionDetail {
    id: string;
    title: string;
    created_at: string;
    updated_at: string;
    messages: Array<{ id: string }>;
    description?: string | null;
    system_prompt?: string | null;
    kg_language?: string | null;
    kg_entity_types?: string[] | null;
}

function mapSessionToWorkspace(session: ServerChatSession): KnowledgeBase {
    return {
        id: session.id,
        name: session.title,
        description: session.description ?? null,
        system_prompt: session.system_prompt ?? null,
        kg_language: session.kg_language ?? null,
        kg_entity_types: session.kg_entity_types ?? null,
        document_count: session.message_count,
        indexed_count: 0,
        created_at: session.created_at,
        updated_at: session.updated_at,
    };
}

export function useWorkspaces() {
    return useQuery({
        queryKey: ["workspaces"],
        queryFn: async () => {
            const sessions = await api.get<ServerChatSession[]>("/chat/sessions");
            return sessions.map(mapSessionToWorkspace);
        },
    });
}

export function useWorkspace(workspaceId: string | null) {
    return useQuery({
        queryKey: ["workspaces", workspaceId],
        queryFn: async () => {
            const session = await api.get<ServerChatSessionDetail>(`/chat/sessions/${workspaceId}`);
            return {
                id: session.id,
                name: session.title,
                description: session.description ?? null,
                system_prompt: session.system_prompt ?? null,
                kg_language: session.kg_language ?? null,
                kg_entity_types: session.kg_entity_types ?? null,
                document_count: session.messages.length,
                indexed_count: 0,
                created_at: session.created_at,
                updated_at: session.updated_at,
            } as KnowledgeBase;
        },
        enabled: !!workspaceId,
    });
}

export function useWorkspaceSummaries() {
    return useQuery({
        queryKey: ["workspaces", "summary"],
        queryFn: async () => {
            const sessions = await api.get<ServerChatSession[]>("/chat/sessions");
            return sessions.map((session) => ({
                id: session.id,
                name: session.title,
                document_count: session.message_count,
            })) as WorkspaceSummary[];
        },
    });
}

export function useCreateWorkspace() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (data: CreateWorkspace) => api.post<ServerChatSession>("/chat/sessions", { title: data.name }).then(mapSessionToWorkspace),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
        },
    });
}

export function useUpdateWorkspace() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: async ({ id, data }: { id: string; data: UpdateWorkspace }) => {
            const session = await api.patch<ServerChatSession>(`/chat/sessions/${id}`, data);
            return mapSessionToWorkspace(session);
        },
        onSuccess: (_updated, variables) => {
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
            queryClient.invalidateQueries({ queryKey: ["workspaces", variables.id] });
        },
    });
}

export function useDeleteWorkspace() {
    const queryClient = useQueryClient();

    return useMutation({
        mutationFn: (id: string) => api.delete(`/chat/sessions/${id}`),
        onSuccess: () => {
            queryClient.invalidateQueries({ queryKey: ["workspaces"] });
        },
    });
}
