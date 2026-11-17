import type { OverlapMatrix as OverlapData } from "../../api/client";

/**
 * Top-k overlap matrix across the 8 conditions (heads ranked by signed score,
 * |A ∩ B| of the top-k sets). Rendered as an accessible table with a light
 * grayscale shading driven by the measured values; the random expectation
 * k*k/144 is shown next to it.
 */

export default function OverlapMatrix({
  data,
  k,
  expectedRandom,
  conditionNames,
}: {
  data: OverlapData;
  k: 5 | 10;
  expectedRandom: number;
  conditionNames: string[];
}) {
  const maxVal = k;
  const shade = (v: number | null) => {
    if (v === null || !Number.isFinite(v)) return "transparent";
    const t = Math.min(1, v / maxVal);
    const g = Math.round(255 - t * 160); // 255 -> 95 (grayscale)
    return `rgb(${g},${g},${g})`;
  };
  const fmt = (v: number | null) => (v === null || !Number.isFinite(v) ? "n/a" : String(v));
  return (
    <div className="table-scroll">
      <table className="kv-table study-table overlap-matrix">
        <caption className="inspector-meta" style={{ textAlign: "left", marginBottom: "0.3rem" }}>
          Top-{k} head overlap |A ∩ B| between conditions (144 heads, ranked by signed score). Random
          expectation k·k/144 = {expectedRandom.toFixed(2)} for k={k}.
        </caption>
        <thead>
          <tr>
            <th scope="col"></th>
            {conditionNames.map((c) => (
              <th scope="col" key={c}>
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {conditionNames.map((rowName, i) => (
            <tr key={rowName}>
              <th scope="row">{rowName}</th>
              {conditionNames.map((colName, j) => {
                const v = data.matrix[i]?.[j] ?? null;
                return (
                  <td
                    key={colName}
                    className="mono"
                    style={{ background: shade(v), color: v !== null && v / maxVal > 0.55 ? "#f2efe6" : undefined }}
                  >
                    {fmt(v)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
