"""Independent reference for the Neural Network Lab's feedforward network.

Reimplements the documented spec from scratch in Python (no shared code with
the TypeScript implementation): mulberry32 PRNG seeded 1234, He-scaled
uniform weights, biases ~ U(-0.1, 0.1), ReLU hidden, stable softmax output.

Usage: python scripts/reference_feedforward.py "0.5,-1,1.2,0"
Prints layer-1 activations and output probabilities for cross-checking the
browser-rendered values.
"""

import sys

SIZES = [4, 6, 6, 5, 3]


def mulberry32(seed: int):
    state = seed & 0xFFFFFFFF

    def rand():
        nonlocal state
        state = (state + 0x6D2B79F5) & 0xFFFFFFFF
        t = state
        t = (t ^ (t >> 15)) * (t | 1) & 0xFFFFFFFF
        t = (t ^ (t + ((t ^ (t >> 7)) * (t | 61) & 0xFFFFFFFF))) & 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296

    return rand


def build(seed: int = 1234):
    rand = mulberry32(seed)
    W, b = [], []
    for l in range(1, len(SIZES)):
        fan_in = SIZES[l - 1]
        scale = (2 / fan_in) ** 0.5
        W.append([[ (rand() * 2 - 1) * scale for _ in range(fan_in)] for _ in range(SIZES[l])])
        b.append([(rand() * 2 - 1) * 0.1 for _ in range(SIZES[l])])
    return W, b


def forward(W, b, x):
    acts = [x]
    zs = []
    for l, (Wl, bl) in enumerate(zip(W, b)):
        z = [sum(w * a for w, a in zip(row, acts[-1])) + bb for row, bb in zip(Wl, bl)]
        zs.append(z)
        if l == len(W) - 1:
            acts.append(z)
        else:
            acts.append([max(0.0, v) for v in z])
    logits = acts[-1]
    m = max(logits)
    exps = [pow(2.718281828459045, v - m) for v in logits]
    s = sum(exps)
    return zs, acts, logits, [e / s for e in exps]


if __name__ == "__main__":
    x = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.5", "-1", "1.2", "0"])]
    W, b = build()
    zs, acts, logits, probs = forward(W, b, x)
    print("layer1 activations:", [round(v, 4) for v in acts[1]])
    print("layer2 activations:", [round(v, 4) for v in acts[2]])
    print("layer3 activations:", [round(v, 4) for v in acts[3]])
    print("logits:", [round(v, 4) for v in logits])
    print("probs:", [round(v, 4) for v in probs])
