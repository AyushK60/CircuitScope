import { useState } from "react";
import { api, type ReproduceSpec } from "../api/client";

/**
 * "Reproduce in Python" control. The server only generates a
 * standalone script; this component downloads it (Blob) or copies it to the
 * clipboard. Runtime text comes from the caller because the script's own
 * header states the measured estimate for its kind.
 */
export default function ReproduceButton({
  spec,
  runtime,
  label = "Reproduce in Python",
  compact = false,
}: {
  spec: ReproduceSpec;
  runtime?: string;
  label?: string;
  compact?: boolean;
}) {
  const [busy, setBusy] = useState<"download" | "copy" | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchScript = async () => {
    const { blob, filename } = await api.reproduce(spec);
    return { blob, filename };
  };

  const download = async () => {
    setBusy("download");
    setError(null);
    setMessage(null);
    try {
      const { blob, filename } = await fetchScript();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
      setMessage(`Downloaded ${filename}`);
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  };

  const copy = async () => {
    setBusy("copy");
    setError(null);
    setMessage(null);
    try {
      const { blob } = await fetchScript();
      const text = await blob.text();
      await navigator.clipboard.writeText(text);
      setMessage("Script copied to the clipboard");
    } catch (e) {
      setError(String((e as Error).message ?? e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className={compact ? "reproduce-control compact" : "reproduce-control"}>
      <div className="reproduce-actions">
        <button onClick={download} disabled={busy !== null} className="primary">
          {busy === "download" ? "Preparing…" : label}
        </button>
        <button onClick={copy} disabled={busy !== null}>
          {busy === "copy" ? "Copying…" : "Copy script"}
        </button>
      </div>
      {runtime && <span className="inspector-meta">{runtime}</span>}
      <p className="inspector-meta" style={{ margin: "0.2rem 0 0" }}>
        The script recomputes these numbers with independent raw HuggingFace code and compares them. It
        does not check the scientific interpretation.
      </p>
      <div aria-live="polite">
        {message && <div className="state-banner ok" role="status">{message}</div>}
        {error && (
          <div className="state-banner error" role="alert">
            Could not create the script: {error}
          </div>
        )}
      </div>
    </div>
  );
}
