import { afterEach, describe, expect, it } from "vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import NeuralLab from "../NeuralLab";

afterEach(cleanup);

describe("Neural Network Lab", () => {
  it("states what it is and describes the network", () => {
    render(<NeuralLab />);
    expect(screen.getByRole("heading", { name: "Neural Network Lab" })).toBeTruthy();
    expect(screen.getByText(/Separate from GPT-2\./)).toBeTruthy();
    expect(screen.getByRole("img", { name: /105 connections/ })).toBeTruthy();
  });

  it("moving an input changes the output probabilities", () => {
    render(<NeuralLab />);
    const before = screen.getAllByText(/%$/).map((n) => n.textContent);
    fireEvent.change(screen.getByLabelText(/in·1/), { target: { value: "-2" } });
    const after = screen.getAllByText(/%$/).map((n) => n.textContent);
    expect(after).not.toEqual(before);
    expect(screen.getByText("-2.0")).toBeTruthy();
  });

  it("shows the real pre-activation of a clicked neuron, and clears on Reset", () => {
    render(<NeuralLab />);
    fireEvent.click(screen.getByRole("button", { name: /Neuron layer 3 index 1/ }));
    expect(screen.getByText("pre-activation z")).toBeTruthy();
    expect(screen.getByText(/ReLU: max\(0, z\)/)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: "Reset" }));
    expect(screen.queryByText("pre-activation z")).toBeNull();
  });

  it("hovering a neuron shows its value and previews the click, without selecting it", () => {
    render(<NeuralLab />);
    const neuron = screen.getByRole("button", { name: /Neuron layer 3 index 1/ });
    fireEvent.mouseEnter(neuron);
    expect(screen.getByText("hidden neuron 1")).toBeTruthy();
    expect(screen.getByText(/^activation /)).toBeTruthy();
    expect(screen.getByText("click to inspect")).toBeTruthy();
    // hovering previews only: the inspector card is still empty
    expect(screen.queryByText("pre-activation z")).toBeNull();
    fireEvent.mouseLeave(neuron);
    expect(screen.queryByText("hidden neuron 1")).toBeNull();
  });

  it("offers the replay and a numbers table for screen readers", () => {
    render(<NeuralLab />);
    expect(screen.getByRole("button", { name: "Replay forward pass" })).toBeTruthy();
    expect(screen.getByText("Accessible activation table")).toBeTruthy();
  });
});
