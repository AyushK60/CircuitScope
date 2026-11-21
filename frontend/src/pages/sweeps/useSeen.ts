import { useEffect, useRef, useState, type RefObject } from "react";

/** True once the element has scrolled into view (and stays true). Without IntersectionObserver
 *  (old browsers, jsdom) it is true from the start, so nothing waits forever. */
export function useSeen<T extends Element>(): [RefObject<T | null>, boolean] {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(() => typeof IntersectionObserver === "undefined");

  useEffect(() => {
    const el = ref.current;
    if (!el || seen) return;
    const io = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) {
          setSeen(true);
          io.disconnect();
        }
      },
      { rootMargin: "0px 0px -15% 0px" },
    );
    io.observe(el);
    return () => io.disconnect();
  }, [seen]);

  return [ref, seen];
}
