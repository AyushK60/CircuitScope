// Typed API client for the CircuitScope backend. All displayed model-derived
// values in the UI come through these calls — no fabricated numbers.

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:8000";

export interface ModelMeta {
  model_id: string;
  revision: string;
  device: string;
  dtype: string;
  n_layers: number;
  n_heads: number;
  d_model: number;
  d_head: number;
  d_mlp: number;
  n_ctx: number;
  d_vocab: number;
  n_params_approx: number;
  tokenizer_class: string;
}

export interface Candidate {
  token: string;
  token_id: number;
  logit: number;
  prob: number;
}

export interface Tokenization {
  tokens: string[];
  token_ids: number[];
  n_tokens: number;
}

export interface Prediction {
  tokens: string[];
  token_ids: number[];
  top_predictions: Candidate[];
  elapsed_seconds: number;
  device: string;
}

export interface FlowBlock {
  layer: number;
  resid_norm_per_token: number[];
  resid_norm_mean: number;
  attention: number[][][];
}

export interface FlowData {
  hook_basis: string;
  tokens: string[];
  token_ids: number[];
  n_layers: number;
  blocks: FlowBlock[];
  top_predictions: Candidate[];
}

export interface AttentionPattern {
  hook: string;
  shape: number[];
  tokens: string[];
  pattern: number[][][];
}

export interface Health {
  status: string;
  model_loaded: boolean;
  model: ModelMeta | null;
}

export interface ExperimentResult {
  config: {
    clean_text: string;
    corrupted_text: string;
    hook: string;
    positions: number[];
    heads: number[] | null;
    candidate_a: string;
    candidate_b: string;
    top_k: number;
  };
  provenance: {
    model_id: string;
    library: string;
    device: string;
    torch: string;
    elapsed_seconds: number;
  };
  tokens_clean: string[];
  tokens_corrupted: string[];
  runs: { clean: Candidate[]; corrupted: Candidate[]; patched: Candidate[] };
  metrics: {
    D_clean: number | null;
    D_corrupted: number | null;
    D_patched: number | null;
    Delta: number | null;
    Recovery: number | null;
    Recovery_note?: string;
    sign_convention: string;
  };
  patch: { hook: string; positions: number[]; heads: number[] | null; source: string };
}

export type ExperimentStatus = "queued" | "running" | "complete" | "failed" | "cancelled";

export interface ExperimentRecord {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: ExperimentResult | null;
  error: string | null;
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => call<Health>("/api/health"),
  predict: (text: string, topK = 8) =>
    call<{ model: ModelMeta; tokenization: Tokenization; prediction: Prediction }>("/api/predict", {
      method: "POST",
      body: JSON.stringify({ text, top_k: topK }),
    }),
  inspect: (text: string, layer: number) =>
    call<{ tokenization: Tokenization; attention?: AttentionPattern; attention_error?: string }>(
      "/api/inspect",
      { method: "POST", body: JSON.stringify({ text, layer }) },
    ),
  flow: (text: string) =>
    call<{ model: ModelMeta } & FlowData>("/api/flow", {
      method: "POST",
      body: JSON.stringify({ text }),
    }),
  startExperiment: (spec: {
    clean_text: string;
    corrupted_text: string;
    hook: string;
    positions: number[];
    heads: number[] | null;
    candidate_a: string;
    candidate_b: string;
  }) =>
    call<{ job_id: string; status: string }>("/api/experiments", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  experiment: (id: string) => call<ExperimentRecord>(`/api/experiments/${id}`),
  cancelExperiment: (id: string) =>
    call<{ job_id: string; cancel_requested: boolean }>(`/api/experiments/${id}/cancel`, {
      method: "POST",
    }),
  exportUrl: (id: string, fmt: "json" | "csv") => `${API_BASE}/api/experiments/${id}/export?fmt=${fmt}`,
  hooks: () => call<{ patchable: string[]; capturable: string[] }>("/api/hooks"),
};
