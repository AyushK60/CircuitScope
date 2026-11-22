import type { FlowData, ModelMeta } from "../api/client";
import RibbonLevel from "./architecture/RibbonLevel";
import BlockLevel from "./architecture/BlockLevel";
import AttentionLevel from "./architecture/AttentionLevel";

/** The full map. Three zoom levels, all drawn from live model metadata:
 *  - model: the ribbon (residual stream) with one card per real block
 *  - block: the seven steps inside one block, in GPT-2's actual order
 *  - attn:  the heads of that block, with a measured focus value each
 *  Measured values (ribbon width, norms, probabilities, focus) appear only when a
 *  real forward pass is loaded. */

export type CanvasLevel = "model" | "block" | "attn";

export default function ArchitectureCanvas({
  meta,
  flow,
  level,
  layer,
  onLevel,
  onLayer,
  onHead,
  intervention,
}: {
  meta: ModelMeta;
  flow: FlowData | null;
  level: CanvasLevel;
  layer: number;
  onLevel: (l: CanvasLevel) => void;
  onLayer: (l: number) => void;
  onHead: (head: number) => void;
  intervention?: { hook: string; position: number } | null;
}) {
  return (
    <div className="canvas-shell">
      {level === "model" && (
        <RibbonLevel
          meta={meta}
          flow={flow}
          layer={layer}
          intervention={intervention}
          onSelectLayer={onLayer}
          onOpenBlock={(l) => {
            onLayer(l);
            onLevel("block");
          }}
          onHead={(l, h) => {
            onLayer(l);
            onHead(h);
          }}
        />
      )}
      {level === "block" && (
        <BlockLevel
          meta={meta}
          flow={flow}
          layer={layer}
          onOpenAttention={() => onLevel("attn")}
          onLayer={onLayer}
          onBack={() => onLevel("model")}
        />
      )}
      {level === "attn" && (
        <AttentionLevel meta={meta} flow={flow} layer={layer} onHead={onHead} onBack={() => onLevel("block")} />
      )}
      <span className="sr-only" aria-live="polite">
        {level === "model" ? `Whole model, block ${layer} selected` : `${level} view of block ${layer}`}
      </span>
    </div>
  );
}
