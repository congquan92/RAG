import type { AgentStep } from "@/types";

const STORAGE_KEY_PREFIX = "nexusrag-agent-steps";
const MAX_MESSAGES_PER_WORKSPACE = 300;

type AgentStepsCacheMap = Record<string, AgentStep[]>;

function getStorageKey(workspaceId: string): string {
    return `${STORAGE_KEY_PREFIX}:${workspaceId}`;
}

function readWorkspaceCache(workspaceId: string): AgentStepsCacheMap {
    if (!workspaceId || typeof window === "undefined") return {};

    try {
        const raw = window.localStorage.getItem(getStorageKey(workspaceId));
        if (!raw) return {};

        const parsed = JSON.parse(raw) as unknown;
        if (!parsed || typeof parsed !== "object") return {};

        const cache: AgentStepsCacheMap = {};
        for (const [messageId, value] of Object.entries(parsed as Record<string, unknown>)) {
            if (!Array.isArray(value)) continue;
            cache[messageId] = value as AgentStep[];
        }
        return cache;
    } catch {
        return {};
    }
}

function writeWorkspaceCache(workspaceId: string, cache: AgentStepsCacheMap): void {
    if (!workspaceId || typeof window === "undefined") return;

    try {
        window.localStorage.setItem(getStorageKey(workspaceId), JSON.stringify(cache));
    } catch {
        // Ignore storage quota errors.
    }
}

export function getCachedAgentSteps(workspaceId: string, messageId: string): AgentStep[] | null {
    if (!workspaceId || !messageId) return null;

    const cache = readWorkspaceCache(workspaceId);
    return cache[messageId] ?? null;
}

export function cacheAgentSteps(workspaceId: string, messageId: string, steps: AgentStep[]): void {
    if (!workspaceId || !messageId || steps.length === 0) return;

    const cache = readWorkspaceCache(workspaceId);
    cache[messageId] = steps;

    const keys = Object.keys(cache);
    if (keys.length > MAX_MESSAGES_PER_WORKSPACE) {
        const overflow = keys.length - MAX_MESSAGES_PER_WORKSPACE;
        for (const key of keys.slice(0, overflow)) {
            delete cache[key];
        }
    }

    writeWorkspaceCache(workspaceId, cache);
}

export function clearCachedAgentSteps(workspaceId: string): void {
    if (!workspaceId || typeof window === "undefined") return;

    try {
        window.localStorage.removeItem(getStorageKey(workspaceId));
    } catch {
        // Ignore storage access issues.
    }
}
