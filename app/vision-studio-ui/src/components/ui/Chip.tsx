import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface ChipProps {
  children: ReactNode;
  variant?: "default" | "primary" | "success" | "warning" | "muted";
  size?: "sm" | "md";
  className?: string;
  icon?: ReactNode;
}

const variantStyles = {
  default: "bg-secondary text-secondary-foreground",
  primary: "bg-primary/10 text-primary",
  success: "bg-success/10 text-success",
  warning: "bg-warning/10 text-warning",
  muted: "bg-muted text-muted-foreground",
};

const sizeStyles = {
  sm: "px-2 py-0.5 text-xs",
  md: "px-3 py-1 text-sm",
};

export function Chip({
  children,
  variant = "default",
  size = "md",
  className,
  icon,
}: ChipProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium rounded-full transition-colors",
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      {children}
    </span>
  );
}
