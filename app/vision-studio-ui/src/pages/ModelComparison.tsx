import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ArrowLeft, FlaskConical, Zap, Clock, Target, HardDrive, Upload, CheckCircle, AlertTriangle } from "lucide-react";
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

const models = [
  { id: "efficientnet-b4", name: "EfficientNet-B4" },
  { id: "resnet-152", name: "ResNet-152" },
  { id: "vit-base", name: "ViT-Base" },
  { id: "densenet-201", name: "DenseNet-201" },
];

const modelMetrics = {
  "efficientnet-b4": { latency: "45ms", f1: "0.94", size: "74MB", confidence: 98.2, prediction: "Malignant" },
  "resnet-152": { latency: "62ms", f1: "0.91", size: "232MB", confidence: 94.7, prediction: "Malignant" },
  "vit-base": { latency: "78ms", f1: "0.93", size: "330MB", confidence: 96.1, prediction: "Malignant" },
  "densenet-201": { latency: "55ms", f1: "0.92", size: "77MB", confidence: 95.8, prediction: "Malignant" },
};

export default function ModelComparison() {
  const navigate = useNavigate();
  const [comparisonMode, setComparisonMode] = useState(true);
  const [modelA, setModelA] = useState("efficientnet-b4");
  const [modelB, setModelB] = useState("resnet-152");
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResults, setHasResults] = useState(false);
  const [uploadedImage, setUploadedImage] = useState<string | null>(null);

  const metricsA = modelMetrics[modelA as keyof typeof modelMetrics];
  const metricsB = modelMetrics[modelB as keyof typeof modelMetrics];
  const modelsAgree = metricsA.prediction === metricsB.prediction;

  const handleAnalyze = () => {
    setIsAnalyzing(true);
    setTimeout(() => {
      setIsAnalyzing(false);
      setHasResults(true);
    }, 2000);
  };

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setUploadedImage(reader.result as string);
        setHasResults(false);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="min-h-screen bg-[#F8FAFC]">
      {/* Sticky Header with Glassmorphism */}
      <header className="sticky top-0 z-50 backdrop-blur-xl bg-white/70 border-b border-slate-200/50">
        <div className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16 sm:h-20">
            {/* Back Button & Logo */}
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

            {/* Comparison Mode Toggle */}
            <div className="flex items-center gap-3 px-4 py-2 rounded-full bg-slate-100/80">
              <Label htmlFor="comparison-mode" className="text-sm font-medium text-slate-600 cursor-pointer">
                Comparison Mode
              </Label>
              <Switch
                id="comparison-mode"
                checked={comparisonMode}
                onCheckedChange={setComparisonMode}
                className="data-[state=checked]:bg-teal-500"
              />
            </div>
          </div>
        </div>
      </header>

      <main className="container max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Control Layer */}
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-10">
          <div className="flex items-center gap-3 w-full sm:w-auto">
            <Select value={modelA} onValueChange={setModelA}>
              <SelectTrigger className="w-full sm:w-52 h-12 rounded-xl bg-white border border-slate-200 shadow-sm font-medium">
                <SelectValue placeholder="Model A" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-slate-200 rounded-xl shadow-lg">
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id} className="rounded-lg cursor-pointer">
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>

            <span className="text-slate-400 font-semibold text-lg">vs</span>

            <Select value={modelB} onValueChange={setModelB}>
              <SelectTrigger className="w-full sm:w-52 h-12 rounded-xl bg-white border border-slate-200 shadow-sm font-medium">
                <SelectValue placeholder="Model B" />
              </SelectTrigger>
              <SelectContent className="bg-white border border-slate-200 rounded-xl shadow-lg">
                {models.map((model) => (
                  <SelectItem key={model.id} value={model.id} className="rounded-lg cursor-pointer">
                    {model.name}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <Button
            onClick={handleAnalyze}
            disabled={!uploadedImage || isAnalyzing}
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

        {/* Image Upload Section */}
        {!uploadedImage && (
          <div className="flex justify-center mb-10">
            <label className="relative cursor-pointer group">
              <div className="w-64 h-64 rounded-2xl border-2 border-dashed border-slate-300 bg-white flex flex-col items-center justify-center gap-4 transition-all duration-300 group-hover:border-teal-400 group-hover:bg-teal-50/30">
                <div className="p-4 rounded-full bg-slate-100 group-hover:bg-teal-100 transition-colors">
                  <Upload className="w-8 h-8 text-slate-400 group-hover:text-teal-500" />
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

        {/* Comparison Grid */}
        {uploadedImage && (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_1fr] gap-6 lg:gap-8 items-start">
            {/* Model A Card */}
            <div className="bg-white rounded-[20px] border border-slate-200/80 shadow-lg p-6 space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-700 font-['Sora',sans-serif]">
                  {models.find(m => m.id === modelA)?.name}
                </h3>
                <span className="text-xs font-medium text-slate-400 bg-slate-100 px-3 py-1 rounded-full">Model A</span>
              </div>

              {hasResults && (
                <div className="space-y-5 animate-fade-in">
                  <div className="text-center py-4">
                    <p className="text-5xl font-bold text-teal-500 font-['Sora',sans-serif]">
                      {metricsA.confidence}%
                    </p>
                    <p className="text-xl font-bold text-slate-800 mt-2 font-['Sora',sans-serif]">
                      {metricsA.prediction}
                    </p>
                  </div>

                  <div className="space-y-3">
                    <ProbabilityBar label="Malignant" value={metricsA.confidence} isHighest={true} />
                    <ProbabilityBar label="Benign" value={100 - metricsA.confidence} isHighest={false} />
                  </div>
                </div>
              )}

              {!hasResults && (
                <div className="flex items-center justify-center py-12 text-slate-400">
                  <p className="text-sm">Click "Sync & Analyze" to see results</p>
                </div>
              )}
            </div>

            {/* Center Image */}
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

              {/* Agreement Status Pill */}
              {hasResults && (
                <div
                  className={`flex items-center gap-2 px-4 py-2 rounded-full shadow-md ${
                    modelsAgree
                      ? "bg-emerald-50 border border-emerald-200"
                      : "bg-amber-50 border border-amber-200"
                  }`}
                >
                  {modelsAgree ? (
                    <>
                      <CheckCircle className="w-4 h-4 text-emerald-500" />
                      <span className="text-sm font-medium text-emerald-700">Models Agree</span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-4 h-4 text-amber-500" />
                      <span className="text-sm font-medium text-amber-700">Models Diverge</span>
                    </>
                  )}
                </div>
              )}
            </div>

            {/* Model B Card */}
            <div className="bg-white rounded-[20px] border border-slate-200/80 shadow-lg p-6 space-y-5">
              <div className="flex items-center justify-between">
                <h3 className="text-lg font-semibold text-slate-700 font-['Sora',sans-serif]">
                  {models.find(m => m.id === modelB)?.name}
                </h3>
                <span className="text-xs font-medium text-slate-400 bg-slate-100 px-3 py-1 rounded-full">Model B</span>
              </div>

              {hasResults && (
                <div className="space-y-5 animate-fade-in">
                  <div className="text-center py-4">
                    <p className="text-5xl font-bold text-teal-500 font-['Sora',sans-serif]">
                      {metricsB.confidence}%
                    </p>
                    <p className="text-xl font-bold text-slate-800 mt-2 font-['Sora',sans-serif]">
                      {metricsB.prediction}
                    </p>
                  </div>

                  <div className="space-y-3">
                    <ProbabilityBar label="Malignant" value={metricsB.confidence} isHighest={true} />
                    <ProbabilityBar label="Benign" value={100 - metricsB.confidence} isHighest={false} />
                  </div>
                </div>
              )}

              {!hasResults && (
                <div className="flex items-center justify-center py-12 text-slate-400">
                  <p className="text-sm">Click "Sync & Analyze" to see results</p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Data Table */}
        {hasResults && (
          <div className="mt-10 bg-white rounded-[20px] border border-slate-200/80 shadow-lg overflow-hidden animate-fade-in">
            <div className="p-6 border-b border-slate-100">
              <h3 className="text-lg font-semibold text-slate-700 font-['Sora',sans-serif]">
                Model Comparison Metrics
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-slate-50/50">
                    <th className="px-6 py-4 text-left text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      Metric
                    </th>
                    <th className="px-6 py-4 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      {models.find(m => m.id === modelA)?.name}
                    </th>
                    <th className="px-6 py-4 text-center text-xs font-semibold text-slate-500 uppercase tracking-wider">
                      {models.find(m => m.id === modelB)?.name}
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100">
                  <tr>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100">
                          <Clock className="w-4 h-4 text-slate-500" />
                        </div>
                        <span className="font-medium text-slate-700">Inference Latency</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsA.latency}</td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsB.latency}</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100">
                          <Target className="w-4 h-4 text-slate-500" />
                        </div>
                        <span className="font-medium text-slate-700">F1-Score</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsA.f1}</td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsB.f1}</td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-slate-100">
                          <HardDrive className="w-4 h-4 text-slate-500" />
                        </div>
                        <span className="font-medium text-slate-700">Model Weight Size</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsA.size}</td>
                    <td className="px-6 py-4 text-center font-semibold text-slate-800">{metricsB.size}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
