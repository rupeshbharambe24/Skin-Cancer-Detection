export interface ModelInfo {
  id: string;
  name: string;
  params: number;
  input_size: number;
  class_names: string[];
  weights_path: string;
  weights_found: boolean;
  test_accuracy: number;
}

export interface PredictionResult {
  model_id: string;
  label: string;
  confidence: number;
  probs: Record<string, number>;
  device: string;
}

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

export async function fetchModels(): Promise<ModelInfo[]> {
  const res = await fetch(`${API_URL}/api/models`);
  if (!res.ok) {
    throw new Error("Failed to load models");
  }
  const data = await res.json();
  return data.models as ModelInfo[];
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Poll the backend until it responds, then return the model list. Used on
 * initial page load so the UI waits out the backend startup window (uvicorn
 * boot + model imports) instead of erroring on the first failed fetch.
 *
 * @param options.timeoutMs total time to keep retrying before giving up
 * @param options.intervalMs delay between attempts
 * @param options.onAttempt callback fired before each attempt (1-based index)
 */
export async function fetchModelsWithRetry(options?: {
  timeoutMs?: number;
  intervalMs?: number;
  onAttempt?: (attempt: number) => void;
}): Promise<ModelInfo[]> {
  const timeoutMs = options?.timeoutMs ?? 60000;
  const intervalMs = options?.intervalMs ?? 1500;
  const deadline = Date.now() + timeoutMs;
  let attempt = 0;
  let lastError: unknown;

  while (Date.now() < deadline) {
    attempt += 1;
    options?.onAttempt?.(attempt);
    try {
      return await fetchModels();
    } catch (err) {
      lastError = err;
      await sleep(intervalMs);
    }
  }

  throw lastError ?? new Error("Backend did not become ready in time");
}

export async function reloadModels(modelId?: string): Promise<void> {
  const form = new FormData();
  if (modelId) {
    form.append("model_id", modelId);
  }
  const res = await fetch(`${API_URL}/api/reload`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    throw new Error("Failed to reload model cache");
  }
}

export async function predictImage(params: {
  modelId: string;
  file: File;
  device: "cpu" | "cuda";
}): Promise<PredictionResult> {
  const form = new FormData();
  form.append("model_id", params.modelId);
  form.append("device", params.device);
  form.append("file", params.file);

  const res = await fetch(`${API_URL}/api/predict`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Prediction failed");
  }
  return (await res.json()) as PredictionResult;
}

export async function compareModels(params: {
  modelIdA: string;
  modelIdB: string;
  file: File;
  device: "cpu" | "cuda";
}): Promise<{ a: PredictionResult; b: PredictionResult }> {
  const form = new FormData();
  form.append("model_id_a", params.modelIdA);
  form.append("model_id_b", params.modelIdB);
  form.append("device", params.device);
  form.append("file", params.file);

  const res = await fetch(`${API_URL}/api/compare`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Comparison failed");
  }
  return (await res.json()) as { a: PredictionResult; b: PredictionResult };
}

export async function compareMultipleModels(params: {
  modelIds: string[];
  file: File;
  device: "cpu" | "cuda";
}): Promise<{ results: PredictionResult[] }> {
  const form = new FormData();
  params.modelIds.forEach((id) => form.append("model_ids", id));
  form.append("device", params.device);
  form.append("file", params.file);

  const res = await fetch(`${API_URL}/api/compare-multi`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || "Comparison failed");
  }
  return (await res.json()) as { results: PredictionResult[] };
}
