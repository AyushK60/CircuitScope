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

export interface CandidateComparison {
  logit_a: number;
  logit_b: number;
  prob_a: number;
  prob_b: number;
  D: number;
  candidate_a: string;
  candidate_b: string;
  sign_convention: string;
}

export interface ContinueStep {
  step: number;
  input_so_far: string;
  chosen: string;
  chosen_id: number;
  chosen_prob: number;
  top_predictions: Candidate[];
}

export interface QuestionScore {
  candidate: string;
  tokens: string[];
  token_ids: number[];
  n_tokens: number;
  total_logprob: number;
  mean_logprob: number;
  first_token_prob: number;
  per_token_logprob: number[];
}

export interface QuestionResult {
  prompt: string;
  context: string;
  question: string;
  candidates: QuestionScore[];
  preferred_by_total_logprob: string;
  method: string;
  length_warning: boolean;
  elapsed_seconds: number;
}

export interface ContinueResult {
  text_start: string;
  text_final: string;
  n_tokens: number;
  steps: ContinueStep[];
  elapsed_seconds: number;
  decoding: string;
}

export interface SavedExperiment {
  id: string;
  created_at: number;
  status: string;
  config: Record<string, unknown>;
  error?: string | null;
}

export interface SavedExperimentFull extends SavedExperiment {
  results: ExperimentResult | null;
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

export interface TaskRecommendedHook {
  hook: string;
  positions: number[];
  heads: number[] | null;
  note?: string;
}

export interface TaskPair {
  id: string;
  clean_text: string;
  corrupted_text: string;
  candidate_a: string;
  candidate_b: string;
  positions_of_interest: Record<string, number>;
  recommended_hooks: TaskRecommendedHook[];
}

export interface TaskPairValidation {
  id: string;
  ok: boolean;
  errors: string[];
  n_tokens_clean: number;
  n_tokens_corrupted: number;
  candidate_a_id: number | null;
  candidate_b_id: number | null;
}

export interface Task {
  id: string;
  name: string;
  description: string;
  source: { citation: string; url: string };
  expected_behavior: string;
  pairs: TaskPair[];
  validation?: { ok: boolean; errors: string[]; pairs: TaskPairValidation[] };
}

export interface TaskSummary {
  id: string;
  name: string;
  description: string;
  n_pairs: number;
}

// sweeps, DLA, logit lens ----

export interface DlaComponent {
  name: string;
  kind: "embed" | "pos_embed" | "head" | "attn_bias" | "mlp" | "ln_final_bias";
  layer: number | null;
  head: number | null;
  value: number;
}

export interface DlaReconciliation {
  sum_of_terms: number;
  model_D: number;
  abs_error: number;
  tolerance: number;
  ok: boolean;
}

export interface DlaResult {
  task_id: string;
  pair_id: string;
  run: "clean" | "corrupted";
  tokens: string[];
  position: number;
  candidate_a: string;
  candidate_b: string;
  D: number;
  components: DlaComponent[];
  reconciliation: DlaReconciliation;
  notes: string[];
  provenance: Record<string, unknown>;
}

export interface LensTopToken {
  token: string;
  token_id: number;
  logit: number;
  prob: number;
}

export interface LensStage {
  stage: string;
  layer: number | null;
  D: number | null;
  prob_a: number | null;
  prob_b: number | null;
  top: LensTopToken[];
}

export interface LensFinalMatches {
  max_abs_logit_diff_candidates: number | null;
  max_abs_logit_diff_full: number;
  top1_agrees: boolean;
  ok: boolean;
  tolerance: number;
}

export interface LensResult {
  tokens: string[];
  position: number;
  candidate_a: string | null;
  candidate_b: string | null;
  stages: LensStage[];
  final_matches_model: LensFinalMatches;
  note: string;
  model_D: number | null;
  provenance: Record<string, unknown>;
}

export interface SweepResult {
  job_type: "sweep";
  mode: "layer_position" | "layer_head";
  task_id: string;
  pair_id: string;
  direction: string;
  tokens_clean: string[];
  tokens_corrupted: string[];
  candidate_a: string;
  candidate_b: string;
  baselines: { D_clean: number; D_corrupted: number; denominator: number; warnings: string[] };
  row_labels: string[];
  col_labels: string[];
  recovery: number[][];
  d_patched: number[][];
  n_runs: number;
  elapsed_seconds: number;
  provenance: Record<string, unknown>;
}

export interface SweepJobStatus {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: SweepResult | null;
  error: string | null;
  type: string;
  progress: number;
  progress_message: string;
}

// study ----

export interface MeanCi {
  mean: number | null;
  ci_low: number | null;
  ci_high: number | null;
  n: number;
  n_invalid: number;
}

export interface Rq1Group {
  signed: MeanCi;
  abs: MeanCi;
  n_components: number;
  low_resolution: boolean;
}

export interface Rq1Block {
  n_prompts: number;
  prompt_ids: string[];
  methods: Record<"atp" | "dla", Record<"heads" | "mlps" | "all", Rq1Group>>;
  null: Record<"heads" | "mlps" | "all", { p2_5: number | null; p97_5: number | null; note: string }>;
  beats_control: Record<"atp" | "dla", Record<"heads" | "mlps" | "all", boolean>>;
  beats_control_rule: string;
}

export interface OverlapMatrix {
  matrix: (number | null)[][];
  ci_low: (number | null)[][];
  ci_high: (number | null)[][];
}

export interface Rq2 {
  condition_names: string[];
  n_per_condition: Record<string, number>;
  spearman: {
    matrix: (number | null)[][];
    ci_low: (number | null)[][];
    ci_high: (number | null)[][];
    note: string;
  };
  topk: {
    k5: OverlapMatrix;
    k10: OverlapMatrix;
    expected_random: { k5: number; k10: number; note?: string };
  };
  prompt_averaged_vectors: Record<string, number[] | null>;
}

export interface KnownHeadsReference {
  task?: string;
  verified_against_paper?: boolean;
  note?: string;
  groups?: { group: string; heads: string[] }[];
  error?: string;
}

export interface StudyAnchors {
  known_heads_reference: KnownHeadsReference;
  ranking_basis: string;
  descriptive_only: boolean;
  mean_ranks_by_group: Record<string, { heads: string[]; ranks: number[]; mean_rank: number | null }> | null;
}

export interface StudyPromptRecord {
  id: string;
  template_id: number;
  names: { A: string; B: string; C: string };
  place: string;
  obj: string;
  s2_role: string;
  clean_text: string;
  candidate_a: string;
  candidate_b: string;
  corruptions: { swap: string | null; third: string | null; resample: string | null };
  s2_position: number;
  end_position: number;
  n_tokens: number;
}

export interface StudyBaseline {
  D_corr?: number;
  P_corr?: number;
  abs_D_diff?: number;
  excluded: boolean;
  missing?: boolean;
}

export interface StudyPerPrompt {
  id: string;
  template_id: number;
  s2_role: string;
  D_clean: number;
  P_clean: number;
  baselines: Record<string, StudyBaseline>;
  rec_d: Record<string, number[] | null>;
  rec_p: Record<string, number[] | null>;
  atp: number[] | null;
  dla: number[] | null;
  abl_zero: number[] | null;
  abl_mean: number[] | null;
}

export interface StudyResult {
  job_type: "study";
  params: { n_prompts: number; seed: number };
  direction: string;
  metrics: Record<string, string>;
  exclusion_rule: string;
  prompt_set: {
    prompts: StudyPromptRecord[];
    rejected_draws: number;
    template_counts: Record<string, number>;
    s2_role_counts: Record<string, number>;
  };
  model_accuracy: number;
  exclusions: Record<string, number>;
  per_prompt: StudyPerPrompt[];
  baselines_per_prompt: Record<string, unknown>[];
  summary: { rq1: { all_prompts: Rq1Block; subset_D_clean_gt0: Rq1Block }; rq2: Rq2; anchors: StudyAnchors };
  provenance: Record<string, unknown>;
  elapsed_seconds: number;
}

export interface StudyJobStatus {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: StudyResult | null;
  error: string | null;
  type: string;
  progress: number;
  progress_message: string;
}

// report + explain ----

export type ReadingLevel = "curious" | "student" | "expert";

export interface ReportClaim {
  id: string;
  level: ReadingLevel;
  text: string;
  claim_type: "measured" | "interpretation" | "caveat";
  evidence: string[];
}

export interface ReportAnalogy {
  id: string;
  title: string;
  text: string;
  breaks_when: string;
}

export interface ReportTrustCheck {
  id: string;
  label: string;
  value: number | string | null;
  tolerance: number | null;
  ok: boolean;
  comparison: string;
}

export interface ReportTrust {
  measured: string;
  against: { label: string; value: unknown; note?: string }[];
  checks: ReportTrustCheck[];
  warnings: string[];
  not_shown: string[];
}

export interface ReportSection {
  id: string;
  title: string;
  question: string;
  claims: ReportClaim[];
  data: Record<string, unknown>;
  evidence_tier: "single prompt, live" | "multi-prompt study, saved";
  analogy: ReportAnalogy | null;
  trust: ReportTrust;
}

export interface ReportResult {
  job_type: "report";
  task_id: string;
  pair_id: string;
  study_job_id: string | null;
  direction: string;
  candidate_a: string;
  candidate_b: string;
  tokens_clean: string[];
  sections: ReportSection[];
  lint: { ok: boolean; curious_readability: Record<string, number> };
  provenance: Record<string, unknown>;
  generated_at: string;
}

export interface ReportJobStatus {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: ReportResult | null;
  error: string | null;
  type: string;
  progress: number;
  progress_message: string;
}

export interface ExplainResponse {
  job_id: string;
  job_type: "sweep" | "study" | "agreement";
  level: ReadingLevel;
  claims: ReportClaim[];
  lint: { ok: boolean };
}

// reproduce, verify, agreement, tour ----

export type ReproduceKind = "intervention" | "sweep" | "study_subset" | "report" | "dla" | "lens";

export interface ReproduceSpec {
  kind: ReproduceKind;
  job_id?: string | null;
  task_id?: string | null;
  pair_id?: string | null;
  run?: "clean" | "corrupted" | null;
  text?: string | null;
  candidate_a?: string | null;
  candidate_b?: string | null;
  mode?: "sample" | "full";
}

export interface AgreementComponent {
  label: string;
  kind: "head" | "mlp";
  layer: number;
  head: number | null;
}

export interface AgreementScatterPoint extends AgreementComponent {
  exact: number;
  atp: number;
  dla: number;
}

export interface AgreementMethodStats {
  pearson_heads: number | null;
  spearman_heads: number | null;
  top10_overlap: number;
  expected_random_top10: number;
  pearson_groups: Record<string, { pearson: number | null; n: number }>;
}

export interface AgreementTopItem {
  rank: number;
  label: string;
  index: number;
  value: number;
}

export interface AgreementResult {
  job_type: "agreement";
  task_id: string;
  pair_id: string;
  direction: string;
  candidate_a: string;
  candidate_b: string;
  labels: AgreementComponent[];
  vectors: { exact: number[]; atp: number[]; dla: number[] };
  scatter: AgreementScatterPoint[];
  stats: {
    atp: AgreementMethodStats;
    dla: AgreementMethodStats;
    procedure: string;
  };
  top10: { exact: AgreementTopItem[]; atp: AgreementTopItem[]; dla: AgreementTopItem[] };
  baselines: { D_clean: number; D_corrupted: number; denominator: number };
  dla_reconciliation: {
    sum_of_terms: number;
    model_D: number;
    abs_error: number;
    tolerance: number;
    ok: boolean;
  };
  notes: string[];
  method_notes: Record<string, string>;
  provenance: Record<string, unknown>;
  elapsed_seconds: number;
}

export interface AgreementJobStatus {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: AgreementResult | null;
  error: string | null;
  type: string;
  progress: number;
  progress_message: string;
}

export interface VerifyResult {
  job_type: "verify";
  job_id: string;
  check_id: string;
  title: string;
  tolerance: number;
  script: string;
  ok: boolean;
  exit_code: number | null;
  max_diff: number | null;
  elapsed_seconds: number;
  stdout_tail: string;
  error: string | null;
  script_sha256?: string;
  git_commit?: string;
  model_revision?: string;
  versions?: Record<string, string>;
  checked_at?: string;
}

export interface VerifyCheckEntry {
  check_id: string;
  title: string;
  what: string;
  tolerance: number;
  runtime: string;
  manual: boolean;
  script: string;
  current_script_sha256: string | null;
  status: "pass" | "fail" | "never run";
  ok: boolean | null;
  stale: boolean;
  result: VerifyResult | null;
}

export interface VerifyStatus {
  checks: VerifyCheckEntry[];
  current_git_commit: string | null;
}

export interface VerifyJobStatus {
  id: string;
  status: ExperimentStatus;
  config?: unknown;
  results: VerifyResult | null;
  error: string | null;
  type: string;
  progress: number;
  progress_message: string;
}

/** Pull a filename out of a Content-Disposition header, with a safe fallback. */
function dispositionFilename(header: string | null, fallback: string): string {
  if (!header) return fallback;
  const match = /filename="?([^";]+)"?/i.exec(header);
  if (!match) return fallback;
  // The backend generates these names; still keep only a bare file name.
  const name = match[1].split(/[\\/]/).pop() ?? "";
  return name || fallback;
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
  predict: (text: string, topK = 8, candidateA?: string, candidateB?: string) =>
    call<{ model: ModelMeta; tokenization: Tokenization; prediction: Prediction; comparison?: CandidateComparison; comparison_error?: string }>(
      "/api/predict",
      {
        method: "POST",
        body: JSON.stringify({
          text,
          top_k: topK,
          ...(candidateA && candidateB ? { candidate_a: candidateA, candidate_b: candidateB } : {}),
        }),
      },
    ),
  continue: (
    text: string,
    nTokens = 5,
    opts?: { mode?: "greedy" | "sample"; temperature?: number; top_k?: number; seed?: number },
  ) =>
    call<{ model: ModelMeta } & ContinueResult>("/api/continue", {
      method: "POST",
      body: JSON.stringify({ text, n_tokens: nTokens, ...opts }),
    }),
  question: (body: { context: string; question: string; candidates: string[]; template?: string }) =>
    call<{ model: ModelMeta } & QuestionResult>("/api/question", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  deleteExperiment: (id: string) =>
    call<{ deleted: boolean }>(`/api/experiments/${id}`, { method: "DELETE" }),
  listExperiments: () =>
    call<{ jobs: string[]; saved: SavedExperiment[] }>("/api/experiments"),
  getSavedExperiment: (id: string) => call<SavedExperimentFull>(`/api/experiments/${id}`),
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
  tasks: () =>
    call<{ tasks: TaskSummary[]; load_errors: { file: string; error: string }[] }>("/api/tasks"),
  task: (id: string) => call<Task>(`/api/tasks/${id}`),
  startSweep: (spec: { task_id: string; pair_id?: string | null; mode: "layer_position" | "layer_head" }) =>
    call<{ job_id: string; status: string }>("/api/sweeps", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  jobStatus: (id: string) =>
    call<SweepJobStatus>(`/api/experiments/${id}`),
  studyStatus: (id: string) =>
    call<StudyJobStatus>(`/api/experiments/${id}`),
  startStudy: (spec: { n_prompts?: number; seed?: number }) =>
    call<{ job_id: string; status: string }>("/api/study", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  dla: (taskId: string, pairId: string | null, run: "clean" | "corrupted") =>
    call<DlaResult>("/api/analysis/dla", {
      method: "POST",
      body: JSON.stringify({ task_id: taskId, pair_id: pairId, run }),
    }),
  logitLens: (text: string, candidateA?: string | null, candidateB?: string | null, topK = 5) =>
    call<LensResult>("/api/analysis/logit_lens", {
      method: "POST",
      body: JSON.stringify({
        text,
        top_k: topK,
        ...(candidateA && candidateB ? { candidate_a: candidateA, candidate_b: candidateB } : {}),
      }),
    }),
  // ---- report and explain ----
  startReport: (spec: { task_id: string; pair_id?: string | null; study_job_id?: string | null }) =>
    call<{ job_id: string; status: string }>("/api/report", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  reportStatus: (id: string) => call<ReportJobStatus>(`/api/experiments/${id}`),
  reportExportUrl: (id: string, level: ReadingLevel) =>
    `${API_BASE}/api/report/${id}/export?fmt=md&level=${level}`,
  explain: (jobId: string, level: ReadingLevel = "curious") =>
    call<ExplainResponse>("/api/explain", {
      method: "POST",
      body: JSON.stringify({ job_id: jobId, level }),
    }),
  // ---- reproduce, verify, agreement ----
  reproduce: async (spec: ReproduceSpec): Promise<{ blob: Blob; filename: string }> => {
    const res = await fetch(`${API_BASE}/api/reproduce`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
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
    const blob = await res.blob();
    return {
      blob,
      filename: dispositionFilename(
        res.headers.get("Content-Disposition"),
        `circuitscope_repro_${spec.kind}.py`,
      ),
    };
  },
  verifyStatus: () => call<VerifyStatus>("/api/verify/status"),
  startVerify: (checkId: string) =>
    call<{ job_id: string; status: string }>("/api/verify", {
      method: "POST",
      body: JSON.stringify({ check_id: checkId }),
    }),
  verifyJob: (id: string) => call<VerifyJobStatus>(`/api/experiments/${id}`),
  startAgreement: (spec: { task_id: string; pair_id?: string | null }) =>
    call<{ job_id: string; status: string }>("/api/agreement", {
      method: "POST",
      body: JSON.stringify(spec),
    }),
  agreementStatus: (id: string) => call<AgreementJobStatus>(`/api/experiments/${id}`),
};
