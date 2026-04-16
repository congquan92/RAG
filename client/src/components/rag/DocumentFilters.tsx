import { memo } from "react";
import { Search } from "lucide-react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import type { DocumentStatus } from "@/types";

type FilterStatus = "all" | DocumentStatus;

const TABS: { value: FilterStatus; label: string }[] = [
  { value: "all", label: "Tất cả" },
  { value: "indexed", label: "Đã chỉ mục" },
  { value: "parsing", label: "Đang xử lý" },
  { value: "failed", label: "Lỗi" },
];

interface DocumentFiltersProps {
  searchQuery: string;
  onSearchChange: (q: string) => void;
  statusFilter: FilterStatus;
  onStatusChange: (s: FilterStatus) => void;
  counts: Record<FilterStatus, number>;
}

export type { FilterStatus };

export const DocumentFilters = memo(function DocumentFilters({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusChange,
  counts,
}: DocumentFiltersProps) {
  return (
    <div className="space-y-2">
      {/* Search */}
      <div className="relative">
        <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground pointer-events-none" />
        <Input
          placeholder="Lọc theo tên..."
          value={searchQuery}
          onChange={(e) => onSearchChange(e.target.value)}
          className="pl-8 h-8 text-sm"
        />
      </div>

      {/* Status tabs */}
      <div className="grid grid-cols-2 gap-1 bg-muted/40 rounded-lg p-1">
        {TABS.map((tab) => {
          const isActive = statusFilter === tab.value;
          // Merge processing-like statuses into the "Processing" tab
          let count = counts[tab.value] ?? 0;
          if (tab.value === "parsing") {
            count = (counts.parsing ?? 0) + (counts.indexing ?? 0) + (counts.processing ?? 0);
          }
          return (
            <button
              key={tab.value}
              onClick={() => onStatusChange(tab.value)}
              className={cn(
                "flex items-center justify-between gap-1 px-2.5 py-1.5 text-[11px] font-medium rounded-md transition-colors",
                isActive
                  ? "bg-card text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              <span className="truncate">{tab.label}</span>
              {
                <span className={cn(
                  "inline-flex min-w-4 h-4 items-center justify-center rounded-full px-1 text-[10px] leading-none",
                  isActive
                    ? "bg-primary/15 text-primary"
                    : "bg-muted text-muted-foreground"
                )}>
                  {count}
                </span>
              }
            </button>
          );
        })}
      </div>
    </div>
  );
});
