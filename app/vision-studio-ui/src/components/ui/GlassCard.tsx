import { cn } from "@/lib/utils";
import { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  title?: string;
  subtitle?: string;
  icon?: ReactNode;
}

export function GlassCard({ children, className, title, subtitle, icon }: GlassCardProps) {
  return (
    <div
      className={cn(
        "glass-card rounded-2xl p-6 transition-all duration-300 hover:shadow-xl",
        className
      )}
    >
      {(title || icon) && (
        <div className="flex items-start gap-3 mb-5">
          {icon && (
            <div className="flex-shrink-0 p-2.5 rounded-xl bg-primary/10 text-primary">
              {icon}
            </div>
          )}
          <div className="flex-1 min-w-0">
            {title && (
              <h3 className="text-lg font-semibold text-foreground tracking-tight">
                {title}
              </h3>
            )}
            {subtitle && (
              <p className="text-sm text-muted-foreground mt-0.5">{subtitle}</p>
            )}
          </div>
        </div>
      )}
      {children}
    </div>
  );
}
