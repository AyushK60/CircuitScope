import type { SavedExperiment } from "../../api/client";

/**
 * History compatibility: the Playground history panel and its compare logic
 * assume intervention-shaped results (`results.runs`, `results.metrics`).
 * A record is an intervention if `config.job_type` is absent (legacy records)
 * or equals "intervention"; anything else (e.g. "sweep") is filtered out of
 * this panel and shown nowhere here. The `/api/experiments` list endpoint
 * itself keeps returning every record unchanged.
 */
export function isInterventionRecord(rec: SavedExperiment): boolean {
  const cfg = rec.config as Record<string, unknown> | null | undefined;
  const jobType = cfg?.job_type;
  return jobType === undefined || jobType === "intervention";
}

export function interventionsOnly(records: SavedExperiment[]): SavedExperiment[] {
  return records.filter((r) => r.status === "complete" && isInterventionRecord(r));
}
