// Pure math for the Neural Network Lab's educational feedforward network.
// Architecture fixed by spec: 4 inputs → 6 → 6 → 5 → 3 outputs.
// Deterministic documented initial weights (He initialization, seed 1234).
// Hidden activation: ReLU. Output: raw logits + numerically stable softmax.
// This is an educational simulation - it is NOT GPT-2 and its outputs are
// abstract classes, not real-world predictions.

export const LAYER_SIZES = [4, 6, 6, 5, 3] as const;

export interface LabNetwork {
  sizes: number[];
  /** W[l][j][i]: weight from neuron i in layer l to neuron j in layer l+1 */
  W: number[][][];
  b: number[][];
}

export interface ForwardResult {
  /** pre-activations per layer (z), layers 1..4 */
  z: number[][];
  /** activations per layer (a), layer 0 = inputs, layer 4 = softmax probs */
  a: number[][];
  /** logits at the output layer */
  logits: number[];
  /** stable softmax of the logits */
  probs: number[];
}

function mulberry32(seed: number) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function createLabNetwork(seed = 1234): LabNetwork {
  const rand = mulberry32(seed);
  const W: number[][][] = [];
  const b: number[][] = [];
  for (let l = 1; l < LAYER_SIZES.length; l++) {
    const fanIn = LAYER_SIZES[l - 1];
    const scale = Math.sqrt(2 / fanIn);
    W.push(
      Array.from({ length: LAYER_SIZES[l] }, () =>
        Array.from({ length: fanIn }, () => (rand() * 2 - 1) * scale),
      ),
    );
    b.push(Array.from({ length: LAYER_SIZES[l] }, () => (rand() * 2 - 1) * 0.1));
  }
  return { sizes: [...LAYER_SIZES], W, b };
}

function stableSoftmax(logits: number[]): number[] {
  const m = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - m));
  const sum = exps.reduce((s, v) => s + v, 0);
  return exps.map((v) => v / sum);
}

export function forwardLab(net: LabNetwork, input: number[]): ForwardResult {
  const z: number[][] = [];
  const a: number[][] = [input];
  for (let l = 0; l < net.W.length; l++) {
    const prev = a[l];
    const zl = net.W[l].map((row, j) => row.reduce((s, w, i) => s + w * prev[i], 0) + net.b[l][j]);
    z.push(zl);
    const isOut = l === net.W.length - 1;
    a.push(zl.map((v) => (isOut ? v : Math.max(0, v)))); // ReLU hidden, raw logits out
  }
  const logits = a[a.length - 1];
  return { z, a, logits, probs: stableSoftmax(logits) };
}

/** total number of connections between adjacent layers (for verification) */
export function edgeCount(net: LabNetwork): number {
  return net.W.reduce((s, m) => s + m.length * m[0].length, 0);
}
