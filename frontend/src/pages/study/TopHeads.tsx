import type { KnownHeadsReference, Rq2 } from "../../api/client";
import { headGroups, topHeads } from "./heads";

/**
 * Top-10 heads per condition (by prompt-averaged signed score, highest
 * first), each row badged with the known Wang et al. 2022 groups the head
 * belongs to (from the reference file shipped with the study results; that
 * file is transcribed from memory and must be verified by a reviewer).
 */

const fmt = (v: number) => (Number.isFinite(v) ? v.toFixed(3) : "n/a");

export default function TopHeads({ rq2, known }: { rq2: Rq2; known: KnownHeadsReference | undefined }) {
  const groups = headGroups(known);
  const conditions = rq2.condition_names;
  return (
    <div className="table-scroll">
      <table className="kv-table study-table">
        <caption className="inspector-meta" style={{ textAlign: "left", marginBottom: "0.3rem" }}>
          Top-10 heads per condition by prompt-averaged signed score (highest first). Badges show membership in
          the Wang et al. 2022 head groups from the reference file
          {known?.verified_against_paper === false ? " (transcribed from memory; reviewer must verify)" : ""}.
        </caption>
        <thead>
          <tr>
            <th scope="col">Condition</th>
            <th scope="col">Top-10 heads (L.H = score)</th>
          </tr>
        </thead>
        <tbody>
          {conditions.map((c) => {
            const vec = rq2.prompt_averaged_vectors[c];
            if (!vec) {
              return (
                <tr key={c}>
                  <th scope="row">{c}</th>
                  <td>no usable prompts for this condition</td>
                </tr>
              );
            }
            const top = topHeads(vec, 10);
            return (
              <tr key={c}>
                <th scope="row">{c}</th>
                <td>
                  <span className="head-chip-row">
                    {top.map((t) => (
                      <span key={t.head} className="head-chip" title={groups[t.head]?.join(", ") || "no known group"}>
                        {t.head} = {fmt(t.value)}
                        {groups[t.head]?.length ? <em className="badge">{groups[t.head].join("/")}</em> : null}
                      </span>
                    ))}
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
