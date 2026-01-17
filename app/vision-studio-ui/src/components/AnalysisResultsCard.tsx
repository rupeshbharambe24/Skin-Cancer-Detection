import { Activity, Sparkles } from "lucide-react";
import { GlassCard } from "@/components/ui/GlassCard";
import { ProbabilityBar } from "@/components/ui/ProbabilityBar";
import { Chip } from "@/components/ui/Chip";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { PredictionResult } from "@/lib/api";

interface AnalysisResultsCardProps {
  hasImage: boolean;
  canAnalyze: boolean;
  isAnalyzing: boolean;
  result: PredictionResult | null;
  error?: string | null;
  onAnalyze: () => void;
}

export function AnalysisResultsCard({
  hasImage,
  canAnalyze,
  isAnalyzing,
  result,
  error,
  onAnalyze,
}: AnalysisResultsCardProps) {
  const probabilities = result
    ? Object.entries(result.probs)
        .map(([label, prob]) => ({ label, probability: prob * 100 }))
        .sort((a, b) => b.probability - a.probability)
    : [];

  const highestResult = probabilities[0];
  const confidence = highestResult
    ? highestResult.probability > 70
      ? "High"
      : highestResult.probability > 40
        ? "Medium"
        : "Low"
    : null;

  return (
    <GlassCard
      title="Analysis & Results"
      subtitle="AI-powered lesion classification"
      icon={<Activity className="w-5 h-5" />}
    >
      <div className="space-y-5">
        <Button
          onClick={onAnalyze}
          disabled={!canAnalyze || isAnalyzing}
          className={cn(
            "w-full h-12 rounded-xl font-semibold text-base transition-all duration-300",
            hasImage && !isAnalyzing
              ? "bg-gradient-to-r from-primary to-primary/80 hover:from-primary/90 hover:to-primary/70 text-primary-foreground shadow-lg shadow-primary/25"
              : "bg-secondary text-muted-foreground"
          )}
        >
          {isAnalyzing ? (
            <>
              <div className="w-5 h-5 mr-2 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              Analyzing...
            </>
          ) : (
            <>
              <Sparkles className="w-5 h-5 mr-2" />
              Analyze Image
            </>
          )}
        </Button>

        {error && (
          <div className="p-3 rounded-xl bg-destructive/10 text-destructive text-sm">
            {error}
          </div>
        )}

        {result && probabilities.length > 0 ? (
          <div className="space-y-5 animate-fade-in">
            <div className="p-4 rounded-2xl bg-accent/50 border border-accent">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-xs font-medium text-muted-foreground uppercase tracking-wider">
                    Predicted Class
                  </p>
                  <p className="text-xl font-bold text-foreground mt-1">
                    {highestResult?.label}
                  </p>
                </div>
                <div className="flex flex-col items-end gap-2">
                  <Chip
                    variant={
                      confidence === "High"
                        ? "success"
                        : confidence === "Medium"
                          ? "warning"
                          : "muted"
                    }
                    size="sm"
                  >
                    {confidence} Confidence
                  </Chip>
                  <span className="text-xs text-muted-foreground">
                    Device: {result.device}
                  </span>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <p className="text-sm font-medium text-muted-foreground">
                Class Probabilities
              </p>
              <div className="space-y-4">
                {probabilities.map((entry, index) => (
                  <ProbabilityBar
                    key={entry.label}
                    label={entry.label}
                    value={entry.probability}
                    isHighest={index === 0}
                  />
                ))}
              </div>
            </div>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center py-12 text-center">
            <div className="p-4 rounded-2xl bg-secondary mb-4">
              <Activity className="w-8 h-8 text-muted-foreground" />
            </div>
            <p className="text-sm text-muted-foreground">
              {hasImage
                ? "Click analyze to classify the lesion"
                : "Upload an image to begin analysis"}
            </p>
          </div>
        )}
      </div>
    </GlassCard>
  );
}
