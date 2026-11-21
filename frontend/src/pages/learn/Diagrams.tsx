import type { ReactNode } from "react";

/* Schematic diagrams for the Learn page. They are labelled as illustrations: the shapes show how an idea
 * works, and no number drawn here is a measurement of the running model. Colours come from the theme. */

const INK = "var(--text-0)";
const MUT = "var(--text-2)";
const LINE = "var(--line-1)";
const PAPER = "var(--bg-2)";
const RED = "var(--accent)";
const BLUE = "var(--measure)";
const MONO = { fontFamily: "var(--font-mono)" } as const;

function Frame({ label, children }: { label: string; children: ReactNode }) {
  return (
    <svg viewBox="0 0 360 150" width="100%" role="img" aria-label={label} style={MONO}>
      {children}
    </svg>
  );
}

function Note({ x = 12, y, children }: { x?: number; y: number; children: ReactNode }) {
  return (
    <text x={x} y={y} fontSize={11} fill={MUT}>
      {children}
    </text>
  );
}

function Tokens() {
  const toks = ["John", " and", " Mary", " went"];
  const ids = [7554, 290, 5335, 1816];
  return (
    <Frame label="A sentence split into four tokens, each with its number">
      <text x={12} y={26} fontSize={11} fill={MUT}>
        text
      </text>
      <text x={12} y={50} fontSize={16} fill={INK} style={{ fontFamily: "var(--font-display)" }}>
        John and Mary went
      </text>
      {toks.map((t, i) => (
        <g key={t} transform={`translate(${12 + i * 84},76)`}>
          <rect width={76} height={34} rx={8} fill={PAPER} stroke={LINE} />
          <text x={38} y={22} fontSize={12} textAnchor="middle" fill={INK}>
            {t.replace(" ", "·")}
          </text>
          <text x={38} y={50} fontSize={11} textAnchor="middle" fill={MUT}>
            id {ids[i]}
          </text>
        </g>
      ))}
      <Note y={140}>· marks a leading space, which belongs to the word</Note>
    </Frame>
  );
}

function Embedding() {
  const rows = ["John", " and", " Mary"];
  return (
    <Frame label="Each token becomes a row of numbers">
      {rows.map((r, ri) => (
        <g key={r}>
          <text x={12} y={38 + ri * 38} fontSize={12} fill={INK}>
            {r.replace(" ", "·")}
          </text>
          {Array.from({ length: 16 }, (_, c) => (
            <rect
              key={c}
              x={74 + c * 17}
              y={22 + ri * 38}
              width={14}
              height={22}
              rx={3}
              fill={(ri * 7 + c * 5) % 6 > 2 ? BLUE : RED}
              opacity={0.25 + ((ri * 3 + c * 7) % 9) / 14}
            />
          ))}
        </g>
      ))}
      <Note x={74} y={140}>
        768 numbers per token (16 drawn)
      </Note>
    </Frame>
  );
}

function Attention() {
  const ws = [0.9, 0.5, 0.25, 0.1];
  const words = ["John", "and", "Mary", "went"];
  return (
    <Frame label="One word looking back at earlier words with different weights">
      <Note y={20}>thicker line = more attention · rows sum to 1</Note>
      {ws.map((w, i) => (
        <g key={i}>
          <path
            d={`M 300 52 C 240 52, ${60 + i * 70} 70, ${60 + i * 70} 104`}
            fill="none"
            stroke={RED}
            strokeWidth={Math.max(1, w * 6)}
            opacity={0.3 + w * 0.65}
            strokeLinecap="round"
          />
          <rect x={28 + i * 70} y={104} width={64} height={26} rx={7} fill={PAPER} stroke={LINE} />
          <text x={60 + i * 70} y={121} fontSize={11} textAnchor="middle" fill={INK}>
            {words[i]}
          </text>
        </g>
      ))}
      <rect x={268} y={36} width={64} height={26} rx={7} fill={INK} />
      <text x={300} y={53} fontSize={11} textAnchor="middle" fill="var(--bg-1)">
        to
      </text>
    </Frame>
  );
}

function Heads() {
  return (
    <Frame label="Twelve attention heads side by side">
      <Note y={22}>12 parallel lanes, computed together</Note>
      {Array.from({ length: 12 }, (_, h) => {
        const fill = 14 + ((h * 5) % 7) * 5;
        return (
          <g key={h}>
            <rect x={14 + h * 28} y={40} width={22} height={52} rx={6} fill={PAPER} stroke={LINE} />
            <rect x={14 + h * 28} y={92 - fill} width={22} height={fill} rx={6} fill={BLUE} opacity={0.3} />
            <text x={25 + h * 28} y={112} fontSize={10} textAnchor="middle" fill={MUT}>
              H{h}
            </text>
          </g>
        );
      })}
      <Note y={140}>each lane learns its own behaviour</Note>
    </Frame>
  );
}

function Residual() {
  return (
    <Frame label="Three blocks each adding into one continuous stream">
      <path d="M 14 88 L 346 88" stroke={BLUE} strokeWidth={3} strokeDasharray="6 5" fill="none" />
      {[64, 180, 296].map((x, i) => (
        <g key={x}>
          <rect x={x - 40} y={30} width={80} height={28} rx={8} fill={PAPER} stroke={LINE} />
          <text x={x} y={48} fontSize={11} textAnchor="middle" fill={INK}>
            block {i}
          </text>
          <path d={`M ${x} 58 L ${x} 82`} stroke={RED} strokeWidth={2} fill="none" />
          <path d={`M ${x - 5} 78 L ${x} 85 L ${x + 5} 78`} stroke={RED} strokeWidth={2} fill="none" />
        </g>
      ))}
      <text x={180} y={124} fontSize={11} textAnchor="middle" fill={MUT}>
        each block adds to the same running stream
      </text>
    </Frame>
  );
}

function Activation() {
  const cols = [50, 130, 210, 290];
  return (
    <Frame label="Intermediate numbers held between the steps of the model">
      {cols.map((x, c) => (
        <g key={x}>
          {Array.from({ length: 5 }, (_, r) => (
            <rect
              key={r}
              x={x - 12}
              y={26 + r * 20}
              width={24}
              height={14}
              rx={4}
              fill={(r + c) % 2 ? BLUE : RED}
              opacity={0.25 + ((r * 3 + c * 5) % 7) / 10}
            />
          ))}
          {c < cols.length - 1 && <path d={`M ${x + 18} 66 L ${x + 62} 66`} stroke={LINE} strokeWidth={2} fill="none" />}
        </g>
      ))}
      <Note y={140}>numbers held between two steps · illustration</Note>
    </Frame>
  );
}

function Logit() {
  const rows: [string, number][] = [
    ["Mary", 0.66],
    ["the", 0.07],
    ["John", 0.05],
    ["her", 0.03],
  ];
  return (
    <Frame label="One score for each candidate next word, turned into probabilities">
      {rows.map(([w, p], i) => (
        <g key={w}>
          <text x={12} y={36 + i * 26} fontSize={12} fill={INK}>
            {w}
          </text>
          <rect x={64} y={24 + i * 26} width={250} height={14} rx={4} fill={PAPER} />
          <rect x={64} y={24 + i * 26} width={(p * 250) / 0.7} height={14} rx={4} fill={i === 0 ? BLUE : LINE} />
        </g>
      ))}
      <Note x={64} y={138}>
        scores become probabilities · example
      </Note>
    </Frame>
  );
}

function Patching() {
  return (
    <Frame label="A clean signal copied into a corrupted run, then the model re-run">
      <rect x={14} y={26} width={120} height={30} rx={8} fill={PAPER} stroke={BLUE} />
      <text x={74} y={46} fontSize={11} textAnchor="middle" fill={INK}>
        clean signal
      </text>
      <rect x={14} y={84} width={120} height={30} rx={8} fill={PAPER} stroke={RED} />
      <text x={74} y={104} fontSize={11} textAnchor="middle" fill={INK}>
        corrupted slot
      </text>
      <path d="M 134 41 C 190 41, 190 99, 140 99" fill="none" stroke={INK} strokeWidth={2} strokeDasharray="5 4" />
      <rect x={226} y={52} width={120} height={46} rx={8} fill={INK} />
      <text x={286} y={72} fontSize={11} textAnchor="middle" fill="var(--bg-1)">
        re-run the model
      </text>
      <text x={286} y={88} fontSize={11} textAnchor="middle" fill="var(--bg-3)">
        and measure
      </text>
      <path d="M 150 99 C 190 99, 200 75, 224 75" fill="none" stroke={MUT} />
    </Frame>
  );
}

function Why() {
  const looked = [0.9, 0.15, 0.6, 0.1];
  const used = [0.2, 0.05, 0.85, 0.1];
  const words = ["John", "and", "Mary", "went"];
  return (
    <Frame label="Looking at a word is different from using it">
      {words.map((w, i) => (
        <g key={w}>
          <text x={48 + i * 78} y={140} fontSize={11} textAnchor="middle" fill={INK}>
            {w}
          </text>
          <rect x={26 + i * 78} y={60 - looked[i] * 40} width={20} height={looked[i] * 40} rx={4} fill={RED} opacity={0.75} />
          <rect x={50 + i * 78} y={120 - used[i] * 40} width={20} height={used[i] * 40} rx={4} fill={BLUE} opacity={0.75} />
        </g>
      ))}
      <text x={12} y={14} fontSize={11} fill={RED}>
        red: looked at
      </text>
      <text x={110} y={14} fontSize={11} fill={BLUE}>
        blue: changes the answer
      </text>
      <path d="M 12 64 L 348 64" stroke={LINE} />
      <Note x={250} y={80}>
        illustration
      </Note>
    </Frame>
  );
}

function Network() {
  return (
    <Frame label="Layers of simple units joined by learned connections">
      {[60, 150, 240, 310].map((x, c) =>
        Array.from({ length: c === 3 ? 2 : 4 }, (_, r) => {
          const n = c === 3 ? 2 : 4;
          const y = 75 + (r - (n - 1) / 2) * 30;
          return (
            <g key={`${c}-${r}`}>
              {c < 3 &&
                Array.from({ length: c === 2 ? 2 : 4 }, (_, k) => {
                  const m = c === 2 ? 2 : 4;
                  const y2 = 75 + (k - (m - 1) / 2) * 30;
                  return <path key={k} d={`M ${x + 8} ${y} L ${[150, 240, 310][c] - 8} ${y2}`} stroke={LINE} strokeWidth={0.8} opacity={0.6} />;
                })}
              <circle cx={x} cy={y} r={8} fill={PAPER} stroke={c === 3 ? RED : LINE} />
            </g>
          );
        }),
      )}
      <Note y={142}>units joined by learned connections</Note>
    </Frame>
  );
}

const DIAGRAMS: Record<string, () => ReactNode> = {
  nn: Network,
  token: Tokens,
  embedding: Embedding,
  attention: Attention,
  head: Heads,
  residual: Residual,
  activation: Activation,
  logit: Logit,
  patching: Patching,
  why: Why,
};

export default function TopicDiagram({ id }: { id: string }) {
  const D = DIAGRAMS[id];
  return D ? <D /> : null;
}
