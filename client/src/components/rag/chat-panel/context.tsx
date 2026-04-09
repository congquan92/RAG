import { createContext, useContext } from "react";
import { useQueryClient } from "@tanstack/react-query";
import type { ChatSourceChunk, Document } from "@/types";

export const WsIdCtx = createContext<string>("");
export const DebugCtx = createContext(false);

// Accumulated sources from all assistant messages in current chat.
// Used as fallback when a response references citation IDs from previous turns.
export const AllSourcesCtx = createContext<ChatSourceChunk[]>([]);

export function useFindDoc(documentId: string): Document | undefined {
    const wsId = useContext(WsIdCtx);
    const queryClient = useQueryClient();
    const docs = queryClient.getQueryData<Document[]>(["documents", wsId]);

    return docs?.find((doc) => doc.id === documentId);
}
