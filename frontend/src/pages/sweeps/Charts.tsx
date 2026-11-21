import type { DlaResult, LensResult } from "../../api/client";

const mix = (side: "pos" | "neg", t: number) =>
  `color-mix(in srgb, var(--div-${side}) ${Math.round(Math.max(0, Math.min(1, t)) * 100)}%, var(--div-zero))`;

/** Components that pushed the answer toward candidate A (right) or B (left), plus a 12 × 12 head grid. */
export function DlaBars({ d }: { d: DlaResult }) {
  const top = [...d.components].sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 12);
  const maxAbs = Math.max(...d.components.map((c) => Math.abs(c.value)), 1e-9);
  const heads = d.components.filter((c) => c.kind === "head");
  const headMax = Math.max(...heads.map((c) => Math.abs(c.value)), 1e-9);
  return (
    <div>
      <div className="sw-dla-scale" aria-hidden>
        <span>pushes toward {d.candidate_b.trim() || "B"}</span>
        <span>pushes toward {d.candidate_a.trim() || "A"}</span>
      </div>
      {top.map((c) => {
        const w = (Math.abs(c.value) / maxAbs) * 50;
        return (
          <div className="sw-dla" key={c.name}>
            <span className="sw-dla-name" title={c.name}>
              {c.name}
            </span>
            <span className="sw-dla-track">
              <i
                style={{
                  left: c.value >= 0 ? "50%" : `${50 - w}%`,
                  width: `${w}%`,
                  background: c.value >= 0 ? "var(--div-pos)" : "var(--div-neg)",
                }}
              />
              <b />
            </span>
            <span className="sw-dla-val">{c.value.toFixed(2)}</span>
          </div>
        );
      })}
      <div className="sw-cap" style={{ marginTop: "1.1rem" }}>
        <span>All {heads.length} attention heads</span>
        <span className="static-badge">rows = blocks · columns = heads</span>
      </div>
      <div className="sw-heads" role="img" aria-label={`Head contributions grid, 12 blocks by 12 heads, ${d.run} prompt`}>
        {Array.from({ length: 12 }, (_, l) => (
          <div className="sw-heads-row" key={l}>
            <span>{l}</span>
            {heads.slice(l * 12, l * 12 + 12).map((c) => (
              <i
                key={c.head}
                title={`L${l}.H${c.head}: ${c.value.toFixed(4)}`}
                style={{ background: mix(c.value >= 0 ? "pos" : "neg", Math.abs(c.value) / headMax) }}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  );
}

/** Top guess after each block: bar height = its probability, blue once it matches the final answer. */
export function LensChart({ lens }: { lens: LensResult }) {
  const finalTok = lens.stages[lens.stages.length - 1]?.top[0]?.token;
  const short = (s: string) => (s === "embeddings" ? "emb" : s.replace(/^block\s*/, ""));
  return (
    <>
      <div
        className="sw-lens"
        role="img"
        aria-label={`Top guess and its probability after the embeddings and after each of the ${lens.stages.length - 1} blocks`}
      >
        {lens.stages.map((s) => {
          const t = s.top[0];
          const p = t?.prob ?? 0;
          const hit = !!t && t.token === finalTok;
          return (
            <div className="sw-lens-col" key={s.stage} title={`${s.stage}: ${t?.token ?? "n/a"} (${p.toFixed(3)})`}>
              <span className="sw-lens-bar">
                <i className={hit ? "hit" : ""} style={{ height: `${Math.max(2, p * 100)}%` }} />
              </span>
              <span className="sw-lens-tok">{t ? t.token.trim() || "␣" : "-"}</span>
              <span className="sw-lens-stage">{short(s.stage)}</span>
            </div>
          );
        })}
      </div>
      <p className="sw-note">Blue = the same word as the final answer. Bar height = how likely the model thinks it is at that point.</p>
    </>
  );
}
