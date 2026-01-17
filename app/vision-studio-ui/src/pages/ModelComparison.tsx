import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import {
  ArrowLeft,
  FlaskConical,
  Zap,
} from "lucide-react";
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
import { compareModels, fetchModels, ModelInfo, PredictionResult } from "@/lib/api";

interface CompareResult {
  a: PredictionResult;
  b: PredictionResult;
}

export default function ModelComparison() {
  const navigate = useNavigate();
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [modelA, setModelA] = useState<string>("");
  const [modelB, setModelB] = useState<string>("");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const [gpuEnabled, setGpuEnabled] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<CompareResult | null>(null);

  useEffect(() => {
    const load = async () => {
      try {
        const data = await fetchModels();
        setModels(data);
        if (!modelA && data[0]) {
          setModelA(data[0].id);
        }
        if (!modelB && data[1]) {
          setModelB(data[1].id);
        }
      } catch {
        setError("Unable to reach the backend. Start the API server and retry.");
      }
    };
    load();
  }, []);

  const modelAInfo = useMemo(
    () => models.find((m) => m.id === modelA),
    [models, modelA]
  );
  const modelBInfo = useMemo(
    () => models.find((m) => m.id === modelB),
    [models, modelB]
  );

  const canAnalyze =
    Boolean(uploadedFile) &&
    Boolean(modelAInfo?.weights_found) &&
    Boolean(modelBInfo?.weights_found);

  const handleAnalyze = async () => {
    if (!uploadedFile || !modelA || !modelB) {
      return;
    }
    if (!modelAInfo?.weights_found || !modelBInfo?.weights_found) {
      setError("Weights missing for one or more selected models.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    setHasResults(false);
    try {
      const response = await compareModels({
        modelIdA: modelA,
        modelIdB: modelB,
        file: uploadedFile,
        device: gpuEnabled ? "cuda" : "cpu",
      });
      setResults(response);
      setHasResults(true);
    } catch {
      setError("Comparison failed. Check the weights and try again.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result as string);
        setUploadedFile(file);
        setHasResults(false);
        setResults(null);
      };
      reader.readAsDataURL(file);
    }
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

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-slate-200/50">
        <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            <div className="flex items-center gap-4">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => navigate("/")}
                className="rounded-xl hover:bg-slate-100"
              >
                <ArrowLeft className="w-5 h-5 text-slate-600" />
              </Button>
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-xl bg-teal-50">
                  <FlaskConical className="w-5 h-5 sm:w-6 sm:h-6 text-teal-500" />
                </div>
                <span className="text-lg sm:text-xl font-bold text-slate-800 tracking-tight font-['Sora',sans-serif]">
                  Lesion Studio
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-slate-100/80">
              <Label htmlFor="gpu-toggle" className="text-sm font-medium text-slate-600 cursor-pointer">
                GPU
              </Label>
              <Switch
                id="gpu-toggle"
                checked={gpuEnabled}
                onCheckedChange={setGpuEnabled}
                className="data-[state=checked]:bg-teal-500"
              />
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {error && (
          <div className="mb-6 rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-10">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Select value={modelA} onValueChange={setModelA}>
              <SelectTrigger className="w-full sm:w-60 h-12 rounded-xl bg-white border border-slate-200 shadow-sm font-medium">
                <SelectValue placeholder="Model A" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-slate-200 rounded-xl shadow-lg">
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id} className="rounded-lg cursor-pointer">
                    {model.test_accuracy.toFixed(2)}% • {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <span className="text-slate-400 font-semibold text-lg">vs</span>

            <Select value={modelB} onValueChange={setModelB}>
              <SelectTrigger className="w-full sm:w-60 h-12 rounded-xl bg-white border border-slate-200 shadow-sm font-medium">
                <SelectValue placeholder="Model B" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-slate-200 rounded-xl shadow-lg">
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id} className="rounded-lg cursor-pointer">
                    {model.test_accuracy.toFixed(2)}% • {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={!canAnalyze || isAnalyzing}
            className="h-12 px-8 rounded-xl bg-gradient-to-r from-teal-500 to-teal-400 hover:from-teal-600 hover:to-teal-500 text-white font-semibold shadow-lg shadow-teal-500/25 transition-all duration-300"
          >
            {isAnalyzing ? (
              <>
                <div className="w-5 h-5 mr-2 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Analyzing...
              </>
            ) : (
              <>
                <Zap className="w-5 h-5 mr-2" />
                Sync & Analyze
              </>
            )}
          </Button>
        </div>

        {!uploadedImage && (
          <div className="flex justify-center mb-10">
            <label className="relative cursor-pointer group">
              <div className="w-64 h-64 rounded-2xl border-2 border-dashed border-slate-300 bg-white flex flex-col items-center justify-center gap-4 transition-all duration-300 group-hover:border-teal-400 group-hover:bg-teal-50/30">
                <div className="p-4 rounded-full bg-slate-100 group-hover:bg-teal-100 transition-colors">
                  <Zap className="w-8 h-8 text-slate-400 group-hover:text-teal-500" />
                </div>
                <div className="text-center">
                  <p className="text-sm font-medium text-slate-700">Upload skin lesion image</p>
                  <p className="text-xs text-slate-400 mt-1">PNG, JPG up to 10MB</p>
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
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 lg:gap-8 items-start">
            <div className="bg-white rounded-[20px] border border-slate-200/80 shadow-lg p-6 space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-700 font-['Sora',sans-serif]">
                  {modelAInfo?.name || "Model A"}
                </h3>
                <span className="text-xs font-medium text-slate-400 bg-slate-100 px-3 py-1 rounded-full">Model A</span>
              </div>

              <div className="flex items-center justify-between text-sm text-slate-500">
                <span>Test Accuracy</span>
                <span className="font-semibold text-slate-700">
                  {modelAInfo ? `${modelAInfo.test_accuracy.toFixed(2)}%` : "--"}
                </span>
              </div>

              {hasResults && results?.a ? (
                <div className="space-y-5 animate-fade-in">
                  <div className="text-center py-4">
                    <p className="text-5xl font-bold text-teal-500 font-['Sora',sans-serif]">
                      {(results.a.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-xl font-bold text-slate-800 mt-2 font-['Sora',sans-serif]">
                      {results.a.label}
                    </p>
                  </div>

                  {renderProbs(results.a)}
                </div>
              ) : (
                <div className="flex items-center justify-center py-12 text-slate-400">
                  <p className="text-sm">Click "Sync & Analyze" to see results</p>
                </div>
              )}
            </div>

            <div className="flex flex-col items-center gap-6 lg:mt-12">
              <div className="relative">
                <div className="w-48 h-48 lg:w-56 lg:h-56 rounded-2xl overflow-hidden border-4 border-white shadow-xl">
                  <img
                    src={uploadedImage}
                    alt="Uploaded lesion"
                    className="w-full h-full object-cover"
                  />
                </div>
                <label className="absolute -bottom-3 left-1/2 -translate-x-1/2 bg-white px-3 py-1.5 rounded-full shadow-md border border-slate-200 text-xs font-medium text-slate-600 cursor-pointer hover:bg-slate-50 transition-colors">
                  Change Image
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </label>
              </div>
            </div>

            <div className="bg-white rounded-[20px] border border-slate-200/80 shadow-lg p-6 space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-700 font-['Sora',sans-serif]">
                  {modelBInfo?.name || "Model B"}
                </h3>
                <span className="text-xs font-medium text-slate-400 bg-slate-100 px-3 py-1 rounded-full">Model B</span>
              </div>

              <div className="flex items-center justify-between text-sm text-slate-500">
                <span>Test Accuracy</span>
                <span className="font-semibold text-slate-700">
                  {modelBInfo ? `${modelBInfo.test_accuracy.toFixed(2)}%` : "--"}
                </span>
              </div>

              {hasResults && results?.b ? (
                <div className="space-y-5 animate-fade-in">
                  <div className="text-center py-4">
                    <p className="text-5xl font-bold text-teal-500 font-['Sora',sans-serif]">
                      {(results.b.confidence * 100).toFixed(1)}%
                    </p>
                    <p className="text-xl font-bold text-slate-800 mt-2 font-['Sora',sans-serif]">
                      {results.b.label}
                    </p>
                  </div>

                  {renderProbs(results.b)}
                </div>
              ) : (
                <div className="flex items-center justify-center py-12 text-slate-400">
                  <p className="text-sm">Click "Sync & Analyze" to see results</p>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
