import { cn } from "@/lib/utils";
import { motion } from "framer-motion";

interface ProbabilityBarProps {
  label: string;
  value: number; // 0-100
  isHighest?: boolean;
  className?: string;
}

export function ProbabilityBar({
  label,
  value,
  isHighest = false,
  className,
}: ProbabilityBarProps) {
  return (
    <div className={cn("space-y-1.5", className)}>
      <div className="flex items-center justify-between">
        <span
          className={cn(
            "text-sm font-medium",
            isHighest ? "text-foreground" : "text-muted-foreground"
          )}
        >
          {label}
        </span>
        <span
          className={cn(
            "text-sm font-semibold tabular-nums",
            isHighest ? "text-primary" : "text-muted-foreground"
          )}
        >
          {value.toFixed(1)}%
        </span>
      </div>
      <div className="h-2 rounded-full bg-secondary overflow-hidden">
        <motion.div
          className={cn(
            "h-full rounded-full",
            isHighest
              ? "bg-gradient-to-r from-primary to-primary/70"
              : "bg-muted-foreground/30"
          )}
          initial={{ width: 0 }}
          animate={{ width: `${value}%` }}
          transition={{ duration: 0.6, ease: "easeOut", delay: 0.1 }}
        />
      </div>
    </div>
  );
}
