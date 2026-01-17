import { cn } from "@/lib/utils";

interface StatusPillProps {
  status: "ready" | "loading" | "error" | "idle";
  className?: string;
}

const statusConfig = {
  ready: {
    label: "Model Ready",
    dotClass: "bg-success",
    bgClass: "bg-success/10",
    textClass: "text-success",
  },
  loading: {
    label: "Loading...",
    dotClass: "bg-warning animate-pulse",
    bgClass: "bg-warning/10",
    textClass: "text-warning",
  },
  error: {
    label: "Error",
    dotClass: "bg-destructive",
    bgClass: "bg-destructive/10",
    textClass: "text-destructive",
  },
  idle: {
    label: "Idle",
    dotClass: "bg-muted-foreground",
    bgClass: "bg-muted",
    textClass: "text-muted-foreground",
  },
};

export function StatusPill({ status, className }: StatusPillProps) {
  const config = statusConfig[status];

  return (
    <div
      className={cn(
        "inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium",
        config.bgClass,
        config.textClass,
        className
      )}
    >
      <span className={cn("w-2 h-2 rounded-full", config.dotClass)} />
      {config.label}
    </div>
  );
}
