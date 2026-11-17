import type { Rq1Block } from "../../api/client";

/**
 * RQ1 agreement table: for each cheap method (AtP, DLA) and component group,
 * the mean per-prompt Spearman rank correlation with exact patching
 * (Rec_D[swap]), the bootstrap 95% CI over prompts, the null 97.5th
 * percentile, and whether the CI lower bound beats that null. All values come
 * verbatim from the study API response.
 */

const fmt = (v: number | null | undefined, d = 3) =>
  v === null || v === undefined || !Number.isFinite(v) ? "n/a" : v.toFixed(d);

export default function Rq1Table({ block }: { block: Rq1Block }) {
  const groups = ["heads", "mlps", "all"] as const;
  const groupLabel = { heads: "heads (144)", mlps: "MLPs (12, low resolution)", all: "all 156" };
  const methods = ["atp", "dla"] as const;
  const methodLabel = { atp: "AtP (attribution patching)", dla: "DLA (direct logit attribution)" };
  return (
    <div className="table-scroll">
      <table className="kv-table study-table">
        <caption className="inspector-meta" style={{ textAlign: "left", marginBottom: "0.3rem" }}>
          Per-prompt Spearman rank correlation with exact patching (Rec_D[swap]); signed values (primary).
          Bootstrap 95% CI over prompts; "beats control" = CI lower bound &gt; null 97.5th percentile.
          {block.n_prompts} prompt(s).
        </caption>
        <thead>
          <tr>
            <th scope="col">Method</th>
            <th scope="col">Components</th>
            <th scope="col">Mean rho (signed)</th>
            <th scope="col">95% CI</th>
            <th scope="col">Mean rho (|abs|)</th>
            <th scope="col">Null 97.5%</th>
            <th scope="col">Beats control</th>
          </tr>
        </thead>
        <tbody>
          {methods.map((m) =>
            groups.map((g) => {
              const grp = block.methods[m][g];
              const beats = block.beats_control[m][g];
              return (
                <tr key={`${m}-${g}`}>
                  <td>{methodLabel[m]}</td>
                  <td>{groupLabel[g]}</td>
                  <td className="mono">{fmt(grp.signed.mean)}</td>
                  <td className="mono">
                    [{fmt(grp.signed.ci_low)}, {fmt(grp.signed.ci_high)}]
                  </td>
                  <td className="mono">{fmt(grp.abs.mean)}</td>
                  <td className="mono">{fmt(block.null[g].p97_5)}</td>
                  <td>{beats ? "yes" : "no"}</td>
                </tr>
              );
            }),
          )}
        </tbody>
      </table>
    </div>
  );
}
