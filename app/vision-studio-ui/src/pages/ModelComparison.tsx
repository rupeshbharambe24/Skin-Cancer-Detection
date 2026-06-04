import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FlaskConical, Zap } from "lucide-react";
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
import { ProbabilityBar } from "@/components/ui/ProbabilityBar";
import { ThemeToggle } from "@/components/ThemeToggle";
import {
  compareMultipleModels,
  fetchModelsWithRetry,
  ModelInfo,
  PredictionResult,
} from "@/lib/api";

const THEME_STORAGE_KEY = "lesion_studio_theme";

export default function ModelComparison() {
  const navigate = useNavigate();
  const [isDark, setIsDark] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelIds, setSelectedModelIds] = useState<string[]>([]);
  const [compareCount, setCompareCount] = useState<number>(2);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [gpuEnabled, setGpuEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResult[]>([]);

  useEffect(() => {
    const savedTheme = localStorage.getItem(THEME_STORAGE_KEY);
    setIsDark(savedTheme === "dark");
  }, []);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem(THEME_STORAGE_KEY, "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem(THEME_STORAGE_KEY, "light");
    }
  }, [isDark]);

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      try {
        const data = await fetchModelsWithRetry();
        if (cancelled) return;
        setModels(data);
        const initialCount = Math.min(Math.max(2, data.length >= 2 ? 2 : data.length), data.length);
        setCompareCount(initialCount);
        setSelectedModelIds(data.slice(0, initialCount).map((m) => m.id));
      } catch {
        if (cancelled) return;
        setError("Unable to reach the backend. Start the API server and retry.");
      }
    };
    load();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!models.length) {
      return;
    }
    const capped = Math.max(2, Math.min(compareCount, models.length));
    if (capped !== compareCount) {
      setCompareCount(capped);
      return;
    }

    setSelectedModelIds((prev) => {
      const next = [...prev];
      while (next.length < capped) {
        const candidate = models.find((m) => !next.includes(m.id));
        if (!candidate) {
          break;
        }
        next.push(candidate.id);
      }
      if (next.length > capped) {
        next.length = capped;
      }
      return next;
    });
  }, [compareCount, models]);

  const selectedInfos = useMemo(
    () =>
      selectedModelIds
        .map((id) => models.find((m) => m.id === id))
        .filter((m): m is ModelInfo => Boolean(m)),
    [models, selectedModelIds]
  );

  const resultMap = useMemo(() => {
    const map = new Map<string, PredictionResult>();
    results.forEach((r) => map.set(r.model_id, r));
    return map;
  }, [results]);

  const hasDuplicateModels = new Set(selectedModelIds).size !== selectedModelIds.length;
  const missingWeightModels = selectedInfos.filter((m) => !m.weights_found).map((m) => m.name);
  const allModelsValid =
    selectedModelIds.length >= 2 &&
    selectedInfos.length === selectedModelIds.length &&
    selectedInfos.every((m) => m.weights_found);

  const canAnalyze = Boolean(uploadedFile) && allModelsValid && !hasDuplicateModels;
  const disabledReason = !uploadedFile
    ? "Upload an image to enable comparison."
    : hasDuplicateModels
      ? "Select unique models for all slots."
      : missingWeightModels.length > 0
        ? `Missing weights for: ${missingWeightModels.join(", ")}`
        : null;

  const handleModelChange = (index: number, modelId: string) => {
    setSelectedModelIds((prev) => {
      const next = [...prev];
      next[index] = modelId;
      return next;
    });
    setError(null);
    setResults([]);
  };

  const handleAnalyze = async () => {
    if (!uploadedFile) {
      return;
    }
    if (hasDuplicateModels) {
      setError("Please select unique models for comparison.");
      return;
    }
    if (!allModelsValid) {
      setError("One or more selected models are invalid or missing weights.");
      return;
    }

    setIsAnalyzing(true);
    setError(null);
    setResults([]);
    try {
      const response = await compareMultipleModels({
        modelIds: selectedModelIds,
        file: uploadedFile,
        device: gpuEnabled ? "cuda" : "cpu",
      });
      setResults(response.results);
    } catch {
      setError("Comparison failed. Check backend and model weights, then retry.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) {
      return;
    }
    const reader = new FileReader();
    reader.onloadend = () => {
      setUploadedImage(reader.result as string);
      setUploadedFile(file);
      setResults([]);
      setError(null);
    };
    reader.readAsDataURL(file);
  };

  const renderProbs = (result?: PredictionResult | null) => {
    if (!result) {
      return null;
    }
    const entries = Object.entries(result.probs)
      .map(([label, prob]) => ({ label, probability: prob * 100 }))
      .sort((a, b) => b.probability - a.probability);
    return (
      <div className="space-y-3">
        {entries.map((entry, index) => (
          <ProbabilityBar
            key={entry.label}
            label={entry.label}
            value={entry.probability}
            isHighest={index === 0}
          />
        ))}
      </div>
    );
  };

  const maxCompareCount = Math.max(2, models.length);

  return (
    <div className="min-h-screen gradient-bg">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-background/70 border-b border-border/50">
        <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/")}
                className="rounded-xl hover:bg-secondary"
              >
                <ArrowLeft className="w-5 h-5 text-muted-foreground" />
              </Button>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-primary/10">
                  <FlaskConical className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
                </div>
                <span className="text-lg sm:text-xl font-bold text-foreground tracking-tight font-['Sora',sans-serif]">
                  Model Comparison
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-secondary/80">
                <Label htmlFor="gpu-toggle" className="text-sm font-medium text-muted-foreground cursor-pointer">
                  GPU
                </Label>
                <Switch
                  id="gpu-toggle"
                  checked={gpuEnabled}
                  onCheckedChange={setGpuEnabled}
                  className="data-[state=checked]:bg-primary"
                />
              </div>
              <ThemeToggle isDark={isDark} onToggle={() => setIsDark(!isDark)} />
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8 space-y-6">
        {error && (
          <div className="rounded-xl border border-destructive/40 bg-destructive/10 px-4 py-3 text-sm text-destructive">
            {error}
          </div>
        )}

        <div className="bg-card rounded-2xl border border-border p-5 sm:p-6 space-y-5 shadow-sm">
          <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-4 items-start">
            <div className="space-y-2">
              <Label className="text-sm text-muted-foreground">Models to compare</Label>
              <Select value={String(compareCount)} onValueChange={(v) => setCompareCount(Number(v))}>
                <SelectTrigger className="h-11 rounded-xl bg-background">
                  <SelectValue placeholder="Select count" />
                </SelectTrigger>
                <SelectContent>
                  {Array.from({ length: maxCompareCount - 1 }, (_, i) => i + 2).map((count) => (
                    <SelectItem key={count} value={String(count)}>
                      {count} models
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {selectedModelIds.map((selectedId, index) => (
                <Select
                  key={`model-slot-${index}`}
                  value={selectedId}
                  onValueChange={(v) => handleModelChange(index, v)}
                >
                  <SelectTrigger className="h-11 rounded-xl bg-background">
                    <SelectValue placeholder={`Model ${index + 1}`} />
                  </SelectTrigger>
                  <SelectContent>
                    {models.map((model) => (
                      <SelectItem key={model.id} value={model.id}>
                        {model.test_accuracy.toFixed(2)}% • {model.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ))}
            </div>
          </div>

          <div className="flex flex-col sm:flex-row sm:items-center gap-4">
            <Button
              onClick={handleAnalyze}
              disabled={!canAnalyze || isAnalyzing}
              className="h-11 px-6 rounded-xl bg-gradient-to-r from-primary to-teal-400 text-white font-semibold"
            >
              {isAnalyzing ? (
                <>
                  <div className="w-5 h-5 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap className="w-5 h-5 mr-2" />
                  Compare Models
                </>
              )}
            </Button>
            {disabledReason && <p className="text-sm text-destructive">{disabledReason}</p>}
          </div>
        </div>

        {!uploadedImage && (
          <div className="flex justify-center">
            <label className="relative cursor-pointer group">
              <div className="w-64 h-64 rounded-2xl border-2 border-dashed border-border bg-card flex flex-col items-center justify-center gap-4 transition-all duration-300 group-hover:border-primary group-hover:bg-accent/40">
                <div className="p-4 rounded-full bg-secondary group-hover:bg-primary/10 transition-colors">
                  <Zap className="w-8 h-8 text-muted-foreground group-hover:text-primary" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-foreground">Upload skin lesion image</p>
                  <p className="text-xs text-muted-foreground mt-1">PNG, JPG up to 10MB</p>
                </div>
              </div>
              <input
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
              />
            </label>
          </div>
        )}

        {uploadedImage && (
          <div className="space-y-6">
            <div className="flex justify-center">
              <div className="relative">
                <div className="w-48 h-48 rounded-2xl overflow-hidden border-2 border-border shadow-md">
                  <img src={uploadedImage} alt="Uploaded lesion" className="w-full h-full object-cover" />
                </div>
                <label className="absolute -bottom-3 left-1/2 -translate-x-1/2 bg-card px-3 py-1.5 rounded-full shadow-md border border-border text-xs font-medium text-muted-foreground cursor-pointer hover:bg-secondary transition-colors">
                  Change Image
                  <input type="file" accept="image/*" onChange={handleImageUpload} className="hidden" />
                </label>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
              {selectedInfos.map((model) => {
                const result = resultMap.get(model.id);
                return (
                  <div key={model.id} className="bg-card rounded-2xl border border-border shadow-sm p-5 space-y-4">
                    <div className="space-y-1">
                      <h3 className="text-base font-semibold text-foreground font-['Sora',sans-serif]">
                        {model.name}
                      </h3>
                      <p className="text-sm text-muted-foreground">
                        Test Accuracy: <span className="font-medium text-foreground">{model.test_accuracy.toFixed(2)}%</span>
                      </p>
                    </div>

                    {result ? (
                      <div className="space-y-4">
                        <div className="text-center py-1">
                          <p className="text-4xl font-bold text-primary font-['Sora',sans-serif]">
                            {(result.confidence * 100).toFixed(1)}%
                          </p>
                          <p className="text-lg font-semibold text-foreground mt-1">{result.label}</p>
                        </div>
                        {renderProbs(result)}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center py-12 text-muted-foreground">
                        <p className="text-sm">Click "Compare Models" to run inference.</p>
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
