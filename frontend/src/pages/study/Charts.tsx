import type { KnownHeadsReference, OverlapMatrix as OverlapData, Rq1Block } from "../../api/client";
import { headGroups, topHeads } from "./heads";
import { GROUP_LABEL } from "./insights";

const f2 = (v: number | null | undefined, d = 2) => (typeof v === "number" && Number.isFinite(v) ? v.toFixed(d) : "n/a");
const mix = (side: "pos" | "neg", t: number) =>
  `color-mix(in srgb, var(--div-${side}) ${Math.round(Math.max(0, Math.min(1, t)) * 100)}%, var(--div-zero))`;

const METHOD = { atp: "Attribution patching", dla: "Direct logit attribution" } as const;
const GROUP = { heads: "144 heads", mlps: "12 MLPs (coarser)", all: "all 156" } as const;

/** One row per method and component group: dot = mean rank agreement with exact patching, bar = 95% range
 *  over prompts, grey tick = the 97.5th percentile of the shuffled control. Filled dot = beats the control. */
export function ForestPlot({ block }: { block: Rq1Block }) {
  const lo = -0.05;
  const hi = 1.05;
  const x = (v: number) => `${((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * 100}%`;
  const rows = (["atp", "dla"] as const).flatMap((m) => (["heads", "mlps", "all"] as const).map((g) => ({ m, g })));
  return (
    <div className="st-forest" role="group" aria-label="Agreement with exact patching, one row per method and component group">
      <div className="st-fr-axis" aria-hidden>
        <span>0 · no agreement</span>
        <span>1 · same ranking</span>
      </div>
      {rows.map(({ m, g }) => {
        const s = block.methods[m][g].signed;
        const beats = block.beats_control[m][g];
        const nullV = block.null[g].p97_5;
        return (
          <div className="st-fr-row" key={`${m}-${g}`}>
            <div className="st-fr-l">
              <b>{METHOD[m]}</b>
              <span>{GROUP[g]}</span>
            </div>
            <div className="st-fr-t" title={`mean ${f2(s.mean, 3)}, 95% range ${f2(s.ci_low, 3)} to ${f2(s.ci_high, 3)}`}>
              {nullV !== null && <i className="st-fr-null" style={{ left: x(nullV) }} title={`control ceiling ${f2(nullV, 3)}`} />}
              {s.ci_low !== null && s.ci_high !== null && (
                <i className={`st-fr-ci ${m}`} style={{ left: x(s.ci_low), width: `calc(${x(s.ci_high)} - ${x(s.ci_low)})` }} />
              )}
              {s.mean !== null && <i className={`st-fr-dot ${m}${beats ? " yes" : ""}`} style={{ left: x(s.mean) }} />}
            </div>
            <div className="st-fr-v">
              {f2(s.mean)}
              <small>{beats ? "beats the control" : "no better than the control"}</small>
            </div>
          </div>
        );
      })}
      <p className="st-key">
        Filled dot = beats the control. Grey tick = what shuffled rankings reach. Bar = 95% range over {block.n_prompts} prompts.
      </p>
    </div>
  );
}

/** 12 × 12 map of the prompt-averaged score of every head; the ten highest are outlined. */
export function HeadsMap({ vector }: { vector: number[] }) {
  const max = Math.max(...vector.map((v) => Math.abs(v)), 1e-9);
  const top = new Set(topHeads(vector, 10).map((t) => t.head));
  return (
    <div className="st-map" role="img" aria-label="Score of each of the 144 attention heads; the ten highest are outlined">
      <span />
      {Array.from({ length: 12 }, (_, i) => (
        <em key={i}>{i}</em>
      ))}
      {Array.from({ length: 12 }, (_, l) => (
        <MapRow key={l} l={l} vector={vector} max={max} top={top} />
      ))}
    </div>
  );
}

function MapRow({ l, vector, max, top }: { l: number; vector: number[]; max: number; top: Set<string> }) {
  return (
    <>
      <em>{l}</em>
      {Array.from({ length: 12 }, (_, k) => {
        const v = vector[l * 12 + k];
        const label = `${l}.${k}`;
        return (
          <b
            key={k}
            className={top.has(label) ? "top" : ""}
            title={`head ${label}: ${f2(v, 3)}`}
            style={{ background: mix(v >= 0 ? "pos" : "neg", Math.abs(v) / max) }}
          />
        );
      })}
    </>
  );
}

/** The ten highest heads, with the published group each one belongs to. */
export function TopList({ vector, known }: { vector: number[] | null; known: KnownHeadsReference | undefined }) {
  const top = topHeads(vector, 10);
  const groups = headGroups(known);
  const max = top[0]?.value || 1;
  if (top.length === 0) return <p className="st-key">No usable prompts for this condition.</p>;
  return (
    <ol className="st-top">
      {top.map((t, i) => (
        <li key={t.head}>
          <span className="st-top-n">{i + 1}</span>
          <span className="st-top-h">head {t.head}</span>
          <span className="st-top-b">
            <i style={{ width: `${Math.max(2, (Math.max(0, t.value) / max) * 100)}%` }} />
          </span>
          <span className="st-top-v">{f2(t.value, 3)}</span>
          <span className="st-top-g">{groups[t.head]?.map((g) => GROUP_LABEL[g] ?? g).join(", ") || "none listed"}</span>
        </li>
      ))}
    </ol>
  );
}

/** How many of the top-k heads two conditions share, as a labelled grid. */
export function OverlapGrid({
  data,
  k,
  names,
  expected,
}: {
  data: OverlapData;
  k: 5 | 10;
  names: string[];
  expected: number;
}) {
  const label = (n: string) => n.replace(/_/g, " ");
  return (
    <div>
      <div
        className="st-ov"
        style={{ gridTemplateColumns: `5.6rem repeat(${names.length}, minmax(0, 1fr))` }}
        role="group"
        aria-label={`Top-${k} heads shared between each pair of conditions`}
      >
        <span />
        {names.map((n) => (
          <em key={n}>{label(n)}</em>
        ))}
        {names.map((rowName, i) => (
          <OverlapRow key={rowName} rowName={label(rowName)} values={data.matrix[i] ?? []} k={k} />
        ))}
      </div>
      <p className="st-key">Shuffled rankings would share about {expected.toFixed(2)} of {k}.</p>
    </div>
  );
}

function OverlapRow({ rowName, values, k }: { rowName: string; values: (number | null)[]; k: number }) {
  return (
    <>
      <em className="r">{rowName}</em>
      {values.map((v, j) => (
        <b
          key={j}
          style={{
            background: v === null ? "transparent" : mix("pos", v / k),
            color: v !== null && v / k > 0.55 ? "var(--bg-1)" : undefined,
          }}
        >
          {v === null ? "n/a" : v}
        </b>
      ))}
    </>
  );
}
