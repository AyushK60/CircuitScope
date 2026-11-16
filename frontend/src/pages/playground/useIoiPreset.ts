import { useEffect } from "react";
import { api } from "../../api/client";

/** Task 3d: load the IOI task's default pair from the registry at mount and
 *  use it as the PRESET source. The rendered UI stays identical — the values
 *  match the hardcoded PRESET — and on any failure the fallback below keeps
 *  the page working offline-from-registry. */
export function useIoiPreset(
  params: { hook?: string; position?: number },
  setters: {
    setClean: (v: string) => void;
    setCorrupted: (v: string) => void;
    setCandA: (v: string) => void;
    setCandB: (v: string) => void;
    setPosition: (updater: (prev: number) => number) => void;
  },
) {
  const { setClean, setCorrupted, setCandA, setCandB, setPosition } = setters;
  useEffect(() => {
    api
      .task("ioi_john_mary")
      .then((t) => {
        const pair = t.pairs.find((p) => p.id === "default") ?? t.pairs[0];
        if (!pair) return;
        setClean(pair.clean_text);
        setCorrupted(pair.corrupted_text);
        setCandA(pair.candidate_a);
        setCandB(pair.candidate_b);
        const s2 = pair.positions_of_interest["second_subject"];
        // don't clobber an Explorer handoff (?position=N)
        if (s2 != null) setPosition((prev) => (params.position != null ? prev : s2));
      })
      .catch(() => {});
    // params comes from useRef(...).current — stable for the page lifetime
  }, [params, setClean, setCorrupted, setCandA, setCandB, setPosition]);
}
