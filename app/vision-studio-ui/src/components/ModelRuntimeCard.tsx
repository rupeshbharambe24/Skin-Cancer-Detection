import { Cpu, RefreshCw, Zap } from "lucide-react";
import { GlassCard } from "@/components/ui/GlassCard";
import { Chip } from "@/components/ui/Chip";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { ModelInfo } from "@/lib/api";

interface ModelRuntimeCardProps {
  models: ModelInfo[];
  selectedModelId: string;
  onModelChange: (modelId: string) => void;
  gpuEnabled: boolean;
  onGpuChange: (value: boolean) => void;
  onReload: () => void;
  isReloading: boolean;
}

export function ModelRuntimeCard({
  models,
  selectedModelId,
  onModelChange,
  gpuEnabled,
  onGpuChange,
  onReload,
  isReloading,
}: ModelRuntimeCardProps) {
  const currentModel = models.find((m) => m.id === selectedModelId);
  const paramsLabel = currentModel
    ? `${(currentModel.params / 1_000_000).toFixed(2)}M`
    : "--";
  const accuracyLabel = currentModel
    ? `${currentModel.test_accuracy.toFixed(2)}%`
    : "--";
  const weightsFile = currentModel?.weights_path?.split(/[/\\]/).pop() ?? "--";

  return (
    <GlassCard
      title="Model & Runtime"
      subtitle="Configure your classification model"
      icon={<Cpu className="w-5 h-5" />}
    >
      <div className="space-y-5">
        <div className="space-y-2">
          <Label className="text-sm font-medium text-muted-foreground">
            Model Architecture
          </Label>
          <Select
            value={selectedModelId}
            onValueChange={onModelChange}
            disabled={models.length === 0}
          >
            <SelectTrigger className="w-full h-11 rounded-xl bg-secondary/50 border-0">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent className="bg-popover border border-border rounded-xl shadow-lg">
              {models.map((model) => (
                <SelectItem key={model.id} value={model.id} className="rounded-lg cursor-pointer">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-primary">
                      {model.test_accuracy.toFixed(2)}%
                    </span>
                    <span>{model.name}</span>
                    <span className="text-xs text-muted-foreground">
                      ({(model.params / 1_000_000).toFixed(2)}M)
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <div className="flex items-center justify-between p-3 rounded-xl bg-secondary/50">
          <div className="flex items-center gap-3">
            <Zap className={`w-4 h-4 ${gpuEnabled ? "text-primary" : "text-muted-foreground"}`} />
            <Label htmlFor="gpu-toggle" className="font-medium cursor-pointer">
              GPU Acceleration
            </Label>
          </div>
          <Switch
            id="gpu-toggle"
            checked={gpuEnabled}
            onCheckedChange={onGpuChange}
          />
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Weights:</span>
          <Chip
            variant={currentModel?.weights_found ? "success" : "warning"}
            size="sm"
          >
            {currentModel?.weights_found ? weightsFile : "Missing"}
          </Chip>
        </div>

        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Model Type:</span>
          <span className="text-sm font-medium text-foreground">Single Model</span>
        </div>

        <div className="space-y-2 pt-2 border-t border-border/50">
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Input Size</span>
            <span className="font-medium text-foreground">
              {currentModel ? `${currentModel.input_size}x${currentModel.input_size}` : "--"}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Classes</span>
            <span className="font-medium text-foreground">
              {currentModel ? currentModel.class_names.length : "--"}
            </span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Parameters</span>
            <span className="font-medium text-foreground">{paramsLabel}</span>
          </div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-muted-foreground">Test Accuracy</span>
            <span className="font-medium text-foreground">{accuracyLabel}</span>
          </div>
        </div>

        <Button
          onClick={onReload}
          disabled={isReloading}
          className="w-full h-11 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-medium"
        >
          <RefreshCw
            className={`w-4 h-4 mr-2 ${isReloading ? "animate-spin" : ""}`}
          />
          {isReloading ? "Reloading Model..." : "Reload Model"}
        </Button>
      </div>
    </GlassCard>
  );
}
