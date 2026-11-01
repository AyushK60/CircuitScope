import { describe, it, expect } from "vitest";
import { createLabNetwork, edgeCount, forwardLab, LAYER_SIZES } from "./feedforward";

describe("feedforward lab network", () => {
  it("has the specified architecture: 4-6-6-5-3", () => {
    const net = createLabNetwork();
    expect(net.sizes).toEqual([4, 6, 6, 5, 3]);
    expect(net.W.length).toBe(4);
    // weight matrix dimensions per layer
    expect(net.W[0].length).toBe(6);
    expect(net.W[0][0].length).toBe(4);
    expect(net.W[1].length).toBe(6);
    expect(net.W[1][0].length).toBe(6);
    expect(net.W[2].length).toBe(5);
    expect(net.W[2][0].length).toBe(6);
    expect(net.W[3].length).toBe(3);
    expect(net.W[3][0].length).toBe(5);
  });

  it("has exactly 4*6 + 6*6 + 6*5 + 5*3 = 105 connections", () => {
    expect(edgeCount(createLabNetwork())).toBe(105);
  });

  it("forward pass matches an independent hand calculation (2-layer slice)", () => {
    const net = createLabNetwork();
    const input = [0.5, -1, 1.2, 0];
    const res = forwardLab(net, input);
    // independent recomputation of layer-1 pre-activations and ReLU
    for (let j = 0; j < 6; j++) {
      let z = net.b[0][j];
      for (let i = 0; i < 4; i++) z += net.W[0][j][i] * input[i];
      expect(res.z[0][j]).toBeCloseTo(z, 10);
      expect(res.a[1][j]).toBeCloseTo(Math.max(0, z), 10);
    }
  });

  it("produces a stable softmax that sums to 1 and is shift-invariant", () => {
    const net = createLabNetwork();
    const r = forwardLab(net, [0.5, -1, 1.2, 0]);
    expect(r.probs.reduce((s, v) => s + v, 0)).toBeCloseTo(1, 10);
    // shift-invariance of softmax is the stability property; verify on raw logits
    const big = r.logits.map((v) => v + 500);
    const m = Math.max(...big);
    const exps = big.map((v) => Math.exp(v - m));
    const sum = exps.reduce((s, v) => s + v, 0);
    const shifted = exps.map((v) => v / sum);
    shifted.forEach((v, i) => expect(v).toBeCloseTo(r.probs[i], 10));
  });

  it("is deterministic for a given seed and input-dependent", () => {
    const net = createLabNetwork(1234);
    const a = forwardLab(net, [1, 0, 0, 0]).probs;
    const b = forwardLab(createLabNetwork(1234), [1, 0, 0, 0]).probs;
    expect(a).toEqual(b);
    const c = forwardLab(net, [0, 1, 0, 0]).probs;
    expect(a).not.toEqual(c);
  });

  it("documents the layer sizes constant", () => {
    expect(LAYER_SIZES).toEqual([4, 6, 6, 5, 3]);
  });
});
