import { useState, useCallback } from "react";
import { Upload, Image as ImageIcon, X } from "lucide-react";
import { GlassCard } from "@/components/ui/GlassCard";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface UploadPreviewCardProps {
  onImageChange?: (file: File | null) => void;
}

export function UploadPreviewCard({ onImageChange }: UploadPreviewCardProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  const setPreviewFromFile = useCallback((file: File) => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
    onImageChange?.(file);
  }, [onImageChange, previewUrl]);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      setPreviewFromFile(file);
    }
  }, [setPreviewFromFile]);

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setPreviewFromFile(file);
    }
  }, [setPreviewFromFile]);

  const handleClear = useCallback(() => {
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }
    setPreviewUrl(null);
    onImageChange?.(null);
  }, [previewUrl, onImageChange]);

  return (
    <GlassCard
      title="Upload & Preview"
      subtitle="Drag and drop or select an image"
      icon={<ImageIcon className="w-5 h-5" />}
    >
      <div className="space-y-4">
        {!previewUrl ? (
          <label
            className={cn(
              "flex flex-col items-center justify-center w-full h-64 rounded-2xl border-2 border-dashed cursor-pointer transition-all duration-300",
              isDragging
                ? "border-primary bg-primary/5 scale-[1.02]"
                : "border-border hover:border-primary/50 hover:bg-secondary/50"
            )}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <input
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileSelect}
            />
            <div className="flex flex-col items-center gap-3 text-center p-6">
              <div className={cn(
                "p-4 rounded-2xl transition-colors",
                isDragging ? "bg-primary/10" : "bg-secondary"
              )}>
                <Upload className={cn(
                  "w-8 h-8 transition-colors",
                  isDragging ? "text-primary" : "text-muted-foreground"
                )} />
              </div>
              <div>
                <p className="text-sm font-medium text-foreground">
                  Drop your image here
                </p>
                <p className="text-xs text-muted-foreground mt-1">
                  or click to browse (PNG, JPG, WEBP)
                </p>
              </div>
            </div>
          </label>
        ) : (
          <div className="relative group">
            <div className="relative w-full h-64 rounded-2xl overflow-hidden bg-secondary">
              <img
                src={previewUrl}
                alt="Preview"
                className="w-full h-full object-contain"
              />
              <div className="absolute inset-0 bg-foreground/0 group-hover:bg-foreground/10 transition-colors" />
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={handleClear}
              className="absolute top-3 right-3 h-8 w-8 rounded-lg bg-background/80 backdrop-blur-sm hover:bg-destructive hover:text-destructive-foreground opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <X className="w-4 h-4" />
            </Button>
          </div>
        )}

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <div className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50" />
          <span>Supported formats: JPEG, PNG, WebP up to 10MB</span>
        </div>
      </div>
    </GlassCard>
  );
}
