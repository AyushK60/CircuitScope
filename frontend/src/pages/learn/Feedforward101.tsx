import { useState } from "react";

/** "Neural Networks 101": a conceptual, fully-connected feedforward example. Deliberately NOT GPT-2: no
 *  token embeddings, no attention, no residual stream. Nothing here is real data; it is a contrast for why
 *  transformers replaced plain dense stacks. */
export default function Feedforward101() {
  const [hoverNode, setHoverNode] = useState<number | null>(null);
  const layers = [
    { n: 3, label: "input", x: 50 },
    { n: 4, label: "hidden 1", x: 150 },
    { n: 4, label: "hidden 2", x: 250 },
    { n: 2, label: "output", x: 350 },
  ];
  const pos = (l: number, i: number) => {
    const layer = layers[l];
    const gap = 44;
    const y0 = 120 - ((layer.n - 1) * gap) / 2;
    return { x: layer.x, y: y0 + i * gap };
  };
  const nodeOwner = (l: number, i: number) => l * 10 + i;
  return (
    <div>
      <svg
        viewBox="0 0 420 240"
        width="100%"
        role="img"
        aria-label="Conceptual feedforward network: input, two hidden layers, output, fully connected"
      >
        {[0, 1, 2].map((l) =>
          Array.from({ length: layers[l].n }, (_, i) =>
            Array.from({ length: layers[l + 1].n }, (_, j) => {
              const a = pos(l, i);
              const b = pos(l + 1, j);
              const mine = hoverNode === nodeOwner(l, i) || hoverNode === nodeOwner(l + 1, j);
              return (
                <path
                  key={`${l}-${i}-${j}`}
                  d={`M ${a.x + 8} ${a.y} L ${b.x - 8} ${b.y}`}
                  stroke={mine ? "var(--accent)" : "var(--line-1)"}
                  strokeWidth={mine ? 1.8 : 0.8}
                  opacity={mine ? 0.9 : 0.45}
                  fill="none"
                />
              );
            }),
          ),
        )}
        {layers.map((layer, l) =>
          Array.from({ length: layer.n }, (_, i) => {
            const p = pos(l, i);
            const mine = hoverNode === nodeOwner(l, i);
            return (
              <circle
                key={`n${l}-${i}`}
                cx={p.x}
                cy={p.y}
                r={9}
                fill={mine ? "var(--accent)" : "var(--bg-2)"}
                stroke={mine ? "var(--accent)" : "var(--line-1)"}
                style={{ cursor: "pointer" }}
                onMouseEnter={() => setHoverNode(nodeOwner(l, i))}
                onMouseLeave={() => setHoverNode(null)}
              />
            );
          }),
        )}
        {layers.map((layer, l) => (
          <text key={l} x={layer.x} y={226} fill="var(--text-2)" fontSize={12} textAnchor="middle">
            {layer.label}
          </text>
        ))}
      </svg>
      <p className="ln-hint">Hover a node to trace its connections. Illustrative structure, no real data.</p>
    </div>
  );
}
