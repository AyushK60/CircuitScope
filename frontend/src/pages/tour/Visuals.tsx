import type { ReportResult, ReportSection, VerifyStatus } from "../../api/client";
import ReproduceButton from "../../components/ReproduceButton";
import { hotWords } from "./copy";
import { sectionOf, type TourScene } from "./scenes";

/** The six pictures of the tour. Every value is read from the report (or the
 *  live verification status); nothing is typed in by hand. */

const num = (x: unknown) => (typeof x === "number" && Number.isFinite(x) ? x : null);
const pct1 = (v: number | null) => (v == null ? "n/a" : `${(v * 100).toFixed(1)}%`);
const dataOf = (s: ReportSection) => s.data as Record<string, unknown>;

/** 1 · a crowd of 100 dots: each dot is one percent of probability */
function AnswerVisual({ section }: { section: ReportSection }) {
  const pred = (dataOf(section).prediction ?? {}) as Record<string, unknown>;
  const a = num(pred.prob_a);
  const b = num(pred.prob_b);
  const na = a == null ? 0 : Math.round(a * 100);
  const nb = b == null ? 0 : Math.min(100 - na, Math.round(b * 100));
  return (
    <div className="tr-answer">
      <div className="tr-crowd" role="img" aria-label={`${na} of 100 dots for ${String(pred.candidate_a)}, ${nb} for ${String(pred.candidate_b)}`}>
        {Array.from({ length: 100 }, (_, i) => (
          <i key={i} className={i < na ? "a" : i < na + nb ? "b" : ""} style={{ ["--i" as string]: i }} />
        ))}
      </div>
      <div className="tr-key">
        <span>
          <em className="a" />
          <b>{pct1(a)}</b> {String(pred.candidate_a ?? "A").trim()}
        </span>
        <span>
          <em className="b" />
          <b>{pct1(b)}</b> {String(pred.candidate_b ?? "B").trim()}
        </span>
      </div>
    </div>
  );
}

/** 2 · the score gap after each stage, with the biggest jump shaded */
function LensVisual({ section }: { section: ReportSection }) {
  const lens = (dataOf(section).lens ?? {}) as { stages?: { stage: string; D: number | null }[] };
  const stages = (lens.stages ?? []).filter((s) => num(s.D) != null);
  if (stages.length < 2) return <p className="tr-cap">No lens readout in this report.</p>;
  const ds = stages.map((s) => s.D as number);
  const w = 560, h = 300, padL = 34, padB = 46;
  const lo = Math.min(...ds, 0), hi = Math.max(...ds, 0);
  const W = w - padL - 14, H = h - padB - 22;
  const x = (i: number) => padL + (i / (ds.length - 1)) * W;
  const y = (v: number) => 18 + (1 - (v - lo) / Math.max(hi - lo, 1e-9)) * H;
  let k = 1, best = -Infinity;
  for (let i = 1; i < ds.length; i++) if (ds[i] - ds[i - 1] > best) { best = ds[i] - ds[i - 1]; k = i; }
  const area = `M ${x(0)},${y(0)} ` + ds.map((d, i) => `L ${x(i).toFixed(1)},${y(d).toFixed(1)}`).join(" ") + ` L ${x(ds.length - 1)},${y(0)} Z`;
  return (
    <svg className="tr-lens" viewBox={`0 0 ${w} ${h}`} role="img" aria-label="Score gap after each stage of the model, with the biggest jump marked">
      <defs>
        <linearGradient id="trlens" x1="0" x2="0" y1="0" y2="1">
          <stop offset="0" stopColor="var(--measure)" stopOpacity="0.4" />
          <stop offset="1" stopColor="var(--measure)" stopOpacity="0" />
        </linearGradient>
      </defs>
      <line x1={padL} x2={w - 14} y1={y(0)} y2={y(0)} stroke="var(--edge)" strokeDasharray="4 4" />
      <text x={padL - 8} y={y(0) + 4} textAnchor="end" fontSize={12} fill="var(--mute)">0</text>
      <rect x={x(k - 1)} y={12} width={x(k) - x(k - 1)} height={H + 12} rx={6} fill="var(--accent)" opacity={0.16} />
      <path d={area} fill="url(#trlens)" className="tr-fade" />
      <polyline pathLength={1} className="tr-draw" points={ds.map((d, i) => `${x(i).toFixed(1)},${y(d).toFixed(1)}`).join(" ")} fill="none" stroke="var(--measure)" strokeWidth={3.4} strokeLinejoin="round" strokeLinecap="round" />
      {ds.map((d, i) => (
        <circle key={i} className="tr-pop" style={{ ["--i" as string]: i }} cx={x(i)} cy={y(d)} r={i === k ? 6.5 : 3.6} fill={i === k ? "var(--accent)" : "var(--measure)"} />
      ))}
      {stages.map((s, i) =>
        i % 2 === 0 || i === k ? (
          <text key={i} x={x(i)} y={h - 26} textAnchor="middle" fontSize={12} fill="var(--mute)">
            {s.stage.replace("embeddings", "start").replace("block ", "")}
          </text>
        ) : null,
      )}
      <text x={padL + W / 2} y={h - 6} textAnchor="middle" fontSize={12} fill="var(--mute)">blocks of the model, in order</text>
      <text x={x(k)} y={y(ds[k]) - 14} textAnchor="middle" fontSize={12} fontWeight={600} fill="var(--accent)">biggest jump</text>
    </svg>
  );
}

/** 3 · the sentence itself, each word lit by how much it carries the answer */
function WordsVisual({ section }: { section: ReportSection }) {
  const sweep = (dataOf(section).sweep ?? {}) as { recovery?: number[][]; tokens?: string[] };
  if (!sweep.recovery || sweep.recovery.length === 0) return <p className="tr-cap">No position sweep in this report.</p>;
  const nCols = Math.max(...sweep.recovery.map((r) => r.length));
  const best = Array.from({ length: nCols }, (_, c) => Math.max(...sweep.recovery!.map((row) => (Number.isFinite(row[c]) ? row[c] : -Infinity))));
  const hot = new Set(hotWords(sweep).map((w) => w.index));
  return (
    <div>
      <p className="tr-sentence">
        {best.map((v, i) => {
          const tok = sweep.tokens?.[i] ?? `#${i}`;
          if (i === 0 && tok.startsWith("<|")) return null;
          const r = Number.isFinite(v) ? Math.max(0, Math.min(1, v)) : 0;
          return (
            <span key={i} className={`tr-w${hot.has(i) ? " hot" : ""}`} style={{ ["--r" as string]: r.toFixed(2), ["--i" as string]: i }} title={`${tok.trim()}: best restoration ${Number.isFinite(v) ? v.toFixed(2) : "n/a"}`}>
              {tok.trim() || tok}
            </span>
          );
        })}
      </p>
      <div className="tr-scale">
        <span>not needed</span>
        <i />
        <span>carries the answer</span>
      </div>
    </div>
  );
}

/** 4 · all 144 attention heads; lit = patching restores the gap, ring = pushes the answer directly */
function PartsVisual({ section }: { section: ReportSection }) {
  const data = dataOf(section);
  const sweep = (data.sweep ?? {}) as { recovery?: number[][] };
  const grid = sweep.recovery;
  const dla = (data.top_dla ?? []) as { layer: number; index: number; label: string }[];
  if (!grid || grid.length === 0) return <p className="tr-cap">No head sweep in this report.</p>;
  const ring = new Set(dla.map((d) => `${d.layer}.${d.index}`));
  return (
    <div className="tr-parts">
      <div className="tr-heads" role="img" aria-label="All attention heads: blocks 0 to 11 from top to bottom, heads 0 to 11 left to right">
        {grid.map((row, l) =>
          row.map((v, h) => {
            const r = Number.isFinite(v) ? Math.max(0, Math.min(1, v / 0.35)) : 0;
            return (
              <i
                key={`${l}.${h}`}
                className={ring.has(`${l}.${h}`) ? "ring" : ""}
                style={{ ["--r" as string]: r.toFixed(2), ["--i" as string]: l + h }}
                title={`head ${l}.${h}: recovery ${Number.isFinite(v) ? v.toFixed(2) : "n/a"}`}
              />
            );
          }),
        )}
      </div>
      <div className="tr-heads-key">
        <span><em className="lit" /> patching restores the gap</span>
        <span><em className="rg" /> pushes the answer directly (top {dla.length})</span>
        <span className="tr-cap">rows: blocks 0 to 11 · columns: heads 0 to 11</span>
      </div>
    </div>
  );
}

type Stat = { pearson_heads?: number | null; top10_overlap?: number; expected_random_top10?: number };

function Gauge({ value, label }: { value: number | null; label: string }) {
  const size = 132, R = size / 2 - 10, C = 2 * Math.PI * R;
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <svg className="tr-gauge" viewBox={`0 0 ${size} ${size}`} role="img" aria-label={`${label}: ${value == null ? "not measured" : value.toFixed(2)}`}>
      <circle cx={size / 2} cy={size / 2} r={R} fill="none" stroke="var(--edge)" strokeWidth={10} />
      <circle className="tr-gauge-arc" cx={size / 2} cy={size / 2} r={R} fill="none" stroke="var(--measure)" strokeWidth={10} strokeLinecap="round" strokeDasharray={`${C * v} ${C}`} transform={`rotate(-90 ${size / 2} ${size / 2})`} />
      <text x="50%" y="52%" textAnchor="middle" className="tr-gauge-n">{value == null ? "n/a" : value.toFixed(2)}</text>
      <text x="50%" y="68%" textAnchor="middle" className="tr-gauge-s">agreement</text>
    </svg>
  );
}

/** 5 · how closely each shortcut matches the exact test */
function MethodsVisual({ section }: { section: ReportSection }) {
  const data = dataOf(section);
  const stats = (data.stats ?? {}) as Record<string, Stat>;
  if (!stats.atp && !stats.dla) return <p className="tr-cap">The agreement measurement was not part of this report.</p>;
  const recon = (data.dla_reconciliation ?? {}) as { abs_error?: number; tolerance?: number; ok?: boolean };
  const cols: { name: string; s: Stat | undefined }[] = [
    { name: "Attribution patching", s: stats.atp },
    { name: "Direct logit attribution", s: stats.dla },
  ];
  return (
    <div>
      <div className="tr-gauges">
        {cols.map(({ name, s }) => (
          <div key={name} className="tr-gcol">
            <Gauge value={s?.pearson_heads ?? null} label={name} />
            <b>{name}</b>
            <div className="tr-dots" aria-label={s?.top10_overlap == null ? "top-10 overlap not measured" : `top-10 overlap ${s.top10_overlap} of 10`}>
              {Array.from({ length: 10 }, (_, i) => (
                <i key={i} className={s?.top10_overlap != null && i < s.top10_overlap ? "on" : ""} style={{ ["--i" as string]: i }} />
              ))}
            </div>
            <span className="tr-cap">
              {s?.top10_overlap == null ? "n/a" : `${s.top10_overlap} of 10`} top heads shared · chance {s?.expected_random_top10 == null ? "n/a" : s.expected_random_top10.toFixed(1)}
            </span>
          </div>
        ))}
      </div>
      {recon.abs_error != null && (
        <p className="tr-cap tr-recon">
          Direct pushes add up to the model's real gap: error <span className="mono">{recon.abs_error.toExponential(1)}</span>, tolerance {recon.tolerance}, {recon.ok ? "ok" : "MISMATCH"}.
        </p>
      )}
    </div>
  );
}

/** 6 · live verification as badges, plus what the tour does not show */
function CheckVisual({ report, jobId, verify }: { report: ReportResult; jobId: string | null; verify: VerifyStatus | null }) {
  const notShown = sectionOf(report, "limits")?.trust.not_shown ?? [];
  return (
    <div className="tr-check">
      <h3>Live verification status</h3>
      {verify ? (
        <ul className="tr-verify">
          {verify.checks.map((c) => {
            const word = c.manual ? "by hand" : c.stale ? "stale" : c.status;
            return (
              <li key={c.check_id} title={c.result?.max_diff != null ? `max diff ${c.result.max_diff.toExponential(2)}, tolerance ${c.tolerance}` : undefined}>
                <span>{c.title}</span>
                <b className={`tr-st ${word.replace(/ /g, "-")}`}>{word}</b>
              </li>
            );
          })}
        </ul>
      ) : (
        <p className="tr-cap">Verification status is unavailable right now.</p>
      )}
      {jobId && <ReproduceButton spec={{ kind: "report", job_id: jobId, mode: "sample" }} runtime="About 30–60 s for --mode sample on CPU" compact />}
      {notShown.length > 0 && (
        <>
          <h3>What this does not show</h3>
          <ul className="tr-notshown">
            {notShown.slice(0, 2).map((s, i) => (
              <li key={i}>{s}</li>
            ))}
          </ul>
          {notShown.length > 2 && <p className="tr-cap">{notShown.length - 2} more in the full report.</p>}
        </>
      )}
    </div>
  );
}

export default function SceneVisual({ scene, report, jobId, verify }: { scene: TourScene; report: ReportResult; jobId: string | null; verify: VerifyStatus | null }) {
  const section = sectionOf(report, scene.sectionId);
  if (scene.visual === "check") return <CheckVisual report={report} jobId={jobId} verify={verify} />;
  if (!section) return <p className="tr-cap">This report has no &quot;{scene.sectionId}&quot; section.</p>;
  if (scene.visual === "answer") return <AnswerVisual section={section} />;
  if (scene.visual === "lens") return <LensVisual section={section} />;
  if (scene.visual === "words") return <WordsVisual section={section} />;
  if (scene.visual === "parts") return <PartsVisual section={section} />;
  return <MethodsVisual section={section} />;
}
