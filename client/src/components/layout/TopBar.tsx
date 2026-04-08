import { memo, useEffect, useState } from "react";
import { useLocation } from "react-router-dom";
import { ChevronRight, Cpu, Database } from "lucide-react";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";

interface ConfigStatus {
    status: string;
    llm_provider: string;
    llm_model: string;
    embedding_model: string;
    embedding_loaded: boolean;
    reranker_loaded: boolean;
    phoenix_enabled: boolean;
}

interface TopBarProps {
    actions?: React.ReactNode;
    className?: string;
}

export const TopBar = memo(function TopBar({ actions, className }: TopBarProps) {
    const location = useLocation();
    const [config, setConfig] = useState<ConfigStatus | null>(null);
    useEffect(() => {
        api.get<ConfigStatus>("/health")
            .then(setConfig)
            .catch(() => {});
    }, []);

    const segments: { label: string; active: boolean }[] = [{ label: "NexusRAG", active: false }];

    if (location.pathname === "/") {
        segments.push({ label: "Knowledge Base", active: true });
    } else if (location.pathname.startsWith("/knowledge-bases/")) {
        segments.push({ label: "Workspace", active: true });
    }

    return (
        <div className={cn("h-12 flex items-center justify-between px-4 border-b border-border flex-shrink-0 bg-background", className)}>
            {/* Breadcrumbs */}
            <div className="flex items-center gap-1.5 text-sm min-w-0">
                {segments.map((seg, i) => (
                    <div key={i} className="flex items-center gap-1.5 min-w-0">
                        {i > 0 && <ChevronRight className="w-3.5 h-3.5 text-muted-foreground flex-shrink-0" />}
                        <span className={cn("truncate", seg.active ? "font-medium text-foreground" : "text-muted-foreground")}>{seg.label}</span>
                    </div>
                ))}
            </div>

            {/* Right-side: model badges + actions */}
            <div className="flex items-center gap-2 flex-shrink-0">
                {config && (
                    <div className="flex items-center gap-1.5">
                        <div
                            className={cn(
                                "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                                config.llm_provider === "ollama" ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-blue-500/10 text-blue-600 dark:text-blue-400",
                            )}
                            title={`Nhà cung cấp LLM: ${config.llm_provider}`}
                        >
                            <Cpu className="w-3 h-3" />
                            <span className="text-muted-foreground/80">LLM</span>
                            <span className="font-semibold">{config.llm_provider}</span>
                        </div>

                        <div
                            className={cn(
                                "flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium",
                                config.llm_provider === "ollama" ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400" : "bg-blue-500/10 text-blue-600 dark:text-blue-400",
                            )}
                            title={`Mô hình LLM: ${config.llm_model}`}
                        >
                            <Cpu className="w-3 h-3" />
                            <span className="text-muted-foreground/80">Model</span>
                            <span className="font-semibold">{config.llm_model}</span>
                        </div>

                        <div
                            className="flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-purple-500/10 text-purple-600 dark:text-purple-400"
                            title={`Embedding: ${config.embedding_model} • ${config.embedding_loaded ? "đã tải" : "đang tải"} • reranker ${config.reranker_loaded ? "sẵn sàng" : "chưa sẵn sàng"}`}
                        >
                            <Database className="w-3 h-3" />
                            <span className="text-muted-foreground/80">Embedding</span>
                            <span className="font-semibold">{config.embedding_model}</span>
                            <span className={cn("w-1.5 h-1.5 rounded-full ml-0.5", config.embedding_loaded && config.reranker_loaded ? "bg-emerald-500" : "bg-amber-500")} />
                        </div>
                    </div>
                )}
                {actions}
            </div>
        </div>
    );
});
