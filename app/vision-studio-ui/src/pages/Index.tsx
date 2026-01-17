import { useEffect, useMemo, useState } from "react";
import { GitCompare, Microscope } from "lucide-react";
import { ThemeToggle } from "@/components/ThemeToggle";
import { StatusPill } from "@/components/ui/StatusPill";
import { ModelRuntimeCard } from "@/components/ModelRuntimeCard";
import { UploadPreviewCard } from "@/components/UploadPreviewCard";
import { AnalysisResultsCard } from "@/components/AnalysisResultsCard";
import { Button } from "@/components/ui/button";
import {
  fetchModels,
  predictImage,
  reloadModels,
  ModelInfo,
  PredictionResult,
} from "@/lib/api";
import { useNavigate } from "react-router-dom";

const Index = () => {
  const navigate = useNavigate();
  const [isDark, setIsDark] = useState(false);
  const [status, setStatus] = useState<"ready" | "loading" | "error" | "idle">("idle");
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModelId, setSelectedModelId] = useState<string>("");
  const [gpuEnabled, setGpuEnabled] = useState(true);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isReloading, setIsReloading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, [isDark]);

  const loadModels = async () => {
    setStatus("loading");
    setError(null);
    try {
      const data = await fetchModels();
      setModels(data);
      setSelectedModelId((prev) => prev || data[0]?.id || "");
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setError("Unable to reach the backend. Start the API server and retry.");
    }
  };

  useEffect(() => {
    loadModels();
  }, []);

  useEffect(() => {
    setResult(null);
    setError(null);
  }, [selectedModelId, selectedFile]);

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId),
    [models, selectedModelId]
  );

  const hasImage = Boolean(selectedFile);
  const canAnalyze = hasImage && Boolean(selectedModel?.weights_found);

  const handleReload = async () => {
    if (!selectedModelId) {
      return;
    }
    setIsReloading(true);
    setStatus("loading");
    setError(null);
    try {
      await reloadModels(selectedModelId);
      await loadModels();
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setError("Failed to reload model cache.");
    } finally {
      setIsReloading(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile || !selectedModelId) {
      return;
    }
    if (!selectedModel?.weights_found) {
      setStatus("error");
      setError("Weights file missing for the selected model.");
      return;
    }
    setIsAnalyzing(true);
    setStatus("loading");
    setError(null);
    try {
      const response = await predictImage({
        modelId: selectedModelId,
        file: selectedFile,
        device: gpuEnabled ? "cuda" : "cpu",
      });
      setResult(response);
      setStatus("ready");
    } catch (err) {
      setStatus("error");
      setError("Prediction failed. Check the model weights and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="min-h-screen gradient-bg">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-background/70 border-b border-border/50">
        <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="p-2 sm:p-2.5 rounded-xl bg-primary/10">
                <Microscope className="w-5 h-5 sm:w-6 sm:h-6 text-primary" />
              </div>
              <div>
                <h1 className="text-lg sm:text-xl font-bold text-foreground tracking-tight">
                  Skin Cancer & Lesion Classifier Studio
                </h1>
                <p className="text-xs sm:text-sm text-muted-foreground hidden sm:block">
                  AI-Powered Dermatological Analysis
                </p>
              </div>
            </div>

            <div className="flex items-center gap-3 sm:gap-4">
              <Button
                onClick={() => navigate("/comparison")}
                variant="outline"
                className="hidden sm:flex items-center gap-2 h-9 px-4 rounded-xl border-primary/30 hover:bg-primary/5 hover:border-primary/50 text-primary font-medium transition-all duration-300"
              >
                <GitCompare className="w-4 h-4" />
                Model Comparison
              </Button>
              <StatusPill status={status} />
              <ThemeToggle isDark={isDark} onToggle={() => setIsDark(!isDark)} />
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-8 lg:py-10">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-5 lg:gap-6">
          <div className="lg:col-span-1">
            <ModelRuntimeCard
              models={models}
              selectedModelId={selectedModelId}
              onModelChange={setSelectedModelId}
              gpuEnabled={gpuEnabled}
              onGpuChange={setGpuEnabled}
              onReload={handleReload}
              isReloading={isReloading}
            />
          </div>

          <div className="lg:col-span-1">
            <UploadPreviewCard onImageChange={setSelectedFile} />
          </div>

          <div className="lg:col-span-1">
            <AnalysisResultsCard
              hasImage={hasImage}
              canAnalyze={canAnalyze}
              isAnalyzing={isAnalyzing}
              result={result}
              error={error}
              onAnalyze={handleAnalyze}
            />
          </div>
        </div>

        <footer className="mt-12 pt-6 border-t border-border/50">
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 text-sm text-muted-foreground">
            <p>Ac 2026 Lesion Classifier Studio. For demonstration purposes only.</p>
            <div className="flex items-center gap-4">
              <span className="hover:text-foreground transition-colors cursor-pointer">
                Documentation
              </span>
              <span className="hover:text-foreground transition-colors cursor-pointer">
                API Reference
              </span>
              <span className="hover:text-foreground transition-colors cursor-pointer">
                Support
              </span>
            </div>
          </div>
        </footer>
      </main>
    </div>
  );
};

export default Index;
