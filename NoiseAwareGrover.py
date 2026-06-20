"""
Week 4 — Noise-aware Grover Experiments
======================================
Builds on a clean Grover implementation and extends it with:
1. Single-target search
2. Multi-target search
3. Exact amplitude-tracking (ideal case)
4. Ideal vs noisy shot-based experiments
5. Iteration sweeps under noise
6. Capstone-oriented analysis for Noise-aware Grover

This script is designed to justify Week 4 of the SoS PoA:
- full Grover implementation and experiments
- single and multi-target search
- noise-model experiments
- first capstone shaping around Noise-aware Grover
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# ─────────────────────────────────────────────
# Configuration data structures
# ─────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    n_qubits: int
    targets: List[int]
    shots: int = 4096
    n_iterations: Optional[int] = None
    label: str = ""


@dataclass
class NoiseConfig:
    p1: float = 0.001
    p2: float = 0.01
    p_meas: float = 0.01
    enabled: bool = True


# ─────────────────────────────────────────────
# Core Grover building blocks
# ─────────────────────────────────────────────

def validate_targets(n_qubits: int, targets: List[int]) -> None:
    N = 2 ** n_qubits
    if not targets:
        raise ValueError("targets must be non-empty")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    bad = [t for t in targets if not (0 <= t < N)]
    if bad:
        raise ValueError(f"targets out of range for {n_qubits} qubits: {bad}")


def bitstring_of(x: int, n_qubits: int) -> str:
    return format(x, f"0{n_qubits}b")


def build_oracle(n_qubits: int, targets: List[int]) -> QuantumCircuit:
    validate_targets(n_qubits, targets)
    qc = QuantumCircuit(n_qubits, name="Oracle")

    for target in targets:
        bits = bitstring_of(target, n_qubits)

        for i, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(i)

        if n_qubits == 1:
            qc.z(0)
        else:
            qc.h(n_qubits - 1)
            qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
            qc.h(n_qubits - 1)

        for i, bit in enumerate(reversed(bits)):
            if bit == "0":
                qc.x(i)

    return qc


def build_diffuser(n_qubits: int) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="Diffuser")
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    if n_qubits == 1:
        qc.z(0)
    else:
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def optimal_iterations(n_qubits: int, n_targets: int = 1) -> int:
    N = 2 ** n_qubits
    return max(1, round((np.pi / 4) * np.sqrt(N / n_targets)))


def grover_angle(n_qubits: int, n_targets: int = 1) -> float:
    N = 2 ** n_qubits
    return np.arcsin(np.sqrt(n_targets / N))


def theoretical_success_probability(n_qubits: int, n_targets: int, k: int) -> float:
    theta = grover_angle(n_qubits, n_targets)
    return float(np.sin((2 * k + 1) * theta) ** 2)


def build_grover_circuit(
    n_qubits: int,
    targets: List[int],
    n_iterations: Optional[int] = None,
    measure: bool = True,
    add_barriers: bool = True,
) -> QuantumCircuit:
    validate_targets(n_qubits, targets)

    if n_iterations is None:
        n_iterations = optimal_iterations(n_qubits, len(targets))

    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c") if measure else None
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    qc.h(range(n_qubits))
    if add_barriers:
        qc.barrier(label="Init |s>")

    oracle = build_oracle(n_qubits, targets)
    diffuser = build_diffuser(n_qubits)

    for i in range(n_iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
        if add_barriers:
            qc.barrier(label=f"Iter {i+1}")

    if measure:
        qc.measure(qr, cr)

    return qc


# ─────────────────────────────────────────────
# Ideal analysis
# ─────────────────────────────────────────────

def exact_target_probability(n_qubits: int, targets: List[int], n_iterations: int) -> float:
    qc = build_grover_circuit(n_qubits, targets, n_iterations=n_iterations, measure=False, add_barriers=False)
    sv = Statevector.from_instruction(qc.decompose().decompose())
    probs = sv.probabilities_dict()
    target_bitstrings = {bitstring_of(t, n_qubits) for t in targets}
    return float(sum(probs.get(bs, 0.0) for bs in target_bitstrings))


def track_amplitudes_single_target(n_qubits: int, target: int, max_iters: Optional[int] = None) -> Dict[str, List[float]]:
    if max_iters is None:
        max_iters = 2 * optimal_iterations(n_qubits, 1)

    target_bs = bitstring_of(target, n_qubits)
    oracle = build_oracle(n_qubits, [target])
    diffuser = build_diffuser(n_qubits)

    init = QuantumCircuit(n_qubits)
    init.h(range(n_qubits))
    sv = Statevector.from_instruction(init)

    empirical = []
    theoretical = []
    non_target = []

    for k in range(max_iters + 1):
        prob_dict = sv.probabilities_dict()
        p = float(prob_dict.get(target_bs, 0.0))
        empirical.append(p)
        non_target.append(1.0 - p)
        theoretical.append(theoretical_success_probability(n_qubits, 1, k))

        if k < max_iters:
            step = QuantumCircuit(n_qubits)
            step.append(oracle, range(n_qubits))
            step.append(diffuser, range(n_qubits))
            sv = sv.evolve(step.decompose().decompose())

    return {
        "target": empirical,
        "non_target": non_target,
        "theory": theoretical,
    }


# ─────────────────────────────────────────────
# Noisy simulation
# ─────────────────────────────────────────────

def build_simple_noise_model(cfg: NoiseConfig) -> NoiseModel:
    noise_model = NoiseModel()
    if not cfg.enabled:
        return noise_model

    err1 = depolarizing_error(cfg.p1, 1)
    err2 = depolarizing_error(cfg.p2, 2)
    ro = ReadoutError([[1 - cfg.p_meas, cfg.p_meas], [cfg.p_meas, 1 - cfg.p_meas]])

    one_qubit_gates = ["x", "h", "z", "sx", "rz"]
    two_qubit_gates = ["cx"]

    for gate in one_qubit_gates:
        noise_model.add_all_qubit_quantum_error(err1, gate)
    for gate in two_qubit_gates:
        noise_model.add_all_qubit_quantum_error(err2, gate)

    noise_model.add_all_qubit_readout_error(ro)
    return noise_model


def run_counts(
    n_qubits: int,
    targets: List[int],
    n_iterations: Optional[int],
    shots: int,
    noise_cfg: Optional[NoiseConfig] = None,
) -> Dict[str, int]:
    qc = build_grover_circuit(n_qubits, targets, n_iterations=n_iterations, measure=True, add_barriers=False)

    if noise_cfg is not None and noise_cfg.enabled:
        sim = AerSimulator(noise_model=build_simple_noise_model(noise_cfg))
    else:
        sim = AerSimulator()

    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts(tqc)


def success_probability(counts: Dict[str, int], targets: List[int], n_qubits: int, shots: int) -> float:
    target_bs = {bitstring_of(t, n_qubits) for t in targets}
    hits = sum(counts.get(bs, 0) for bs in target_bs)
    return hits / shots


def iteration_sweep(
    n_qubits: int,
    targets: List[int],
    max_iters: int,
    shots: int,
    noise_cfg: Optional[NoiseConfig] = None,
) -> List[Dict[str, float]]:
    rows = []
    for k in range(max_iters + 1):
        theory = theoretical_success_probability(n_qubits, len(targets), k)
        exact = exact_target_probability(n_qubits, targets, k)
        counts = run_counts(n_qubits, targets, n_iterations=k, shots=shots, noise_cfg=noise_cfg)
        sampled = success_probability(counts, targets, n_qubits, shots)
        rows.append({
            "iteration": k,
            "theory": theory,
            "exact": exact,
            "sampled": sampled,
        })
    return rows


# ─────────────────────────────────────────────
# Resource metrics
# ─────────────────────────────────────────────

def circuit_metrics(qc: QuantumCircuit) -> Dict[str, int]:
    tqc = transpile(qc.decompose().decompose(), basis_gates=["x", "h", "z", "cx", "rz", "sx"], optimization_level=1)
    ops = tqc.count_ops()
    two_qubit = int(ops.get("cx", 0))
    one_qubit = sum(int(v) for k, v in ops.items() if k != "cx" and k != "measure" and k != "barrier")
    return {
        "depth": int(tqc.depth()),
        "width": int(tqc.num_qubits),
        "cx_count": two_qubit,
        "one_qubit_count": int(one_qubit),
    }


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_amplitude_tracking(n_qubits: int, target: int, out_file: str) -> None:
    k_opt = optimal_iterations(n_qubits, 1)
    max_iters = max(2 * k_opt, 6)
    data = track_amplitudes_single_target(n_qubits, target, max_iters=max_iters)

    xs = list(range(max_iters + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, data["target"], "o-", label="Exact statevector", color="#2563eb", linewidth=2)
    ax.plot(xs, data["theory"], "--", label=r"Theory $\sin^2((2k+1)\theta)$", color="#dc2626", linewidth=2)
    ax.axvline(k_opt, color="#111827", linestyle=":", label=f"Optimal k={k_opt}")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Target probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(f"Grover amplitude amplification (n={n_qubits}, target={target})")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_iteration_sweep(ideal_rows: List[Dict[str, float]], noisy_rows: List[Dict[str, float]], title: str, out_file: str) -> None:
    xs = [r["iteration"] for r in ideal_rows]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, [r["theory"] for r in ideal_rows], "--", color="#111827", linewidth=2, label="Theory")
    ax.plot(xs, [r["sampled"] for r in ideal_rows], "o-", color="#2563eb", linewidth=2, label="Ideal sampled")
    ax.plot(xs, [r["sampled"] for r in noisy_rows], "s-", color="#dc2626", linewidth=2, label="Noisy sampled")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Success probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_noise_strength_scan(noise_levels: List[float], probs: List[float], out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(noise_levels, probs, "o-", color="#7c3aed", linewidth=2)
    ax.set_xlabel("Two-qubit depolarizing probability p2")
    ax.set_ylabel("Success probability at optimal iteration")
    ax.grid(alpha=0.25)
    ax.set_title("Noise-aware Grover: success degradation with noise strength")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Experiment runners
# ─────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def print_counts_summary(counts: Dict[str, int], top_n: int = 8) -> None:
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    for bs, c in ranked:
        print(f"  {bs}: {c}")


def run_single_and_multi_target_experiments() -> None:
    print_header("WEEK 4 — FULL GROVER EXPERIMENTS")

    experiments = [
        ExperimentConfig(n_qubits=4, targets=[11], shots=4096, label="Single-target (4 qubits, target=11)"),
        ExperimentConfig(n_qubits=4, targets=[3, 11], shots=4096, label="Multi-target (4 qubits, targets=3,11)"),
        ExperimentConfig(n_qubits=5, targets=[6], shots=4096, label="Single-target (5 qubits, target=6)"),
    ]

    base_noise = NoiseConfig(p1=0.001, p2=0.01, p_meas=0.01, enabled=True)

    for exp in experiments:
        k_opt = optimal_iterations(exp.n_qubits, len(exp.targets)) if exp.n_iterations is None else exp.n_iterations
        N = 2 ** exp.n_qubits
        target_labels = [bitstring_of(t, exp.n_qubits) for t in exp.targets]
        print(f"\nExperiment: {exp.label}")
        print(f"  Search space N = {N}")
        print(f"  Targets = {exp.targets} -> {target_labels}")
        print(f"  Optimal iterations = {k_opt}")
        print(f"  Classical avg queries ~ {N/2:.1f}")
        print(f"  Grover query scale ~ {(np.pi/4)*np.sqrt(N/len(exp.targets)):.3f}")

        ideal_counts = run_counts(exp.n_qubits, exp.targets, k_opt, exp.shots, noise_cfg=None)
        ideal_p = success_probability(ideal_counts, exp.targets, exp.n_qubits, exp.shots)
        print(f"  Ideal success probability = {ideal_p:.4f}")
        print("  Top ideal counts:")
        print_counts_summary(ideal_counts)

        noisy_counts = run_counts(exp.n_qubits, exp.targets, k_opt, exp.shots, noise_cfg=base_noise)
        noisy_p = success_probability(noisy_counts, exp.targets, exp.n_qubits, exp.shots)
        print(f"  Noisy success probability = {noisy_p:.4f}")
        print("  Top noisy counts:")
        print_counts_summary(noisy_counts)

        qc = build_grover_circuit(exp.n_qubits, exp.targets, k_opt, measure=True, add_barriers=False)
        metrics = circuit_metrics(qc)
        print("  Circuit metrics:")
        print(f"    depth = {metrics['depth']}")
        print(f"    width = {metrics['width']}")
        print(f"    CX count = {metrics['cx_count']}")
        print(f"    1q gate count = {metrics['one_qubit_count']}")


def run_capstone_oriented_noise_sweeps() -> None:
    print_header("NOISE-AWARE GROVER SWEEPS")

    n_qubits = 4
    target = 11
    targets = [target]
    shots = 4096
    k_opt = optimal_iterations(n_qubits, 1)
    max_iters = max(2 * k_opt, 6)

    ideal_rows = iteration_sweep(n_qubits, targets, max_iters=max_iters, shots=shots, noise_cfg=None)
    noisy_cfg = NoiseConfig(p1=0.001, p2=0.01, p_meas=0.01, enabled=True)
    noisy_rows = iteration_sweep(n_qubits, targets, max_iters=max_iters, shots=shots, noise_cfg=noisy_cfg)

    print("\nIteration sweep (ideal vs noisy):")
    print(f" {'k':>3} {'Theory':>10} {'Ideal':>10} {'Noisy':>10}")
    print(" " + "-" * 39)
    for i, j in zip(ideal_rows, noisy_rows):
        print(f" {i['iteration']:>3} {i['theory']:>10.4f} {i['sampled']:>10.4f} {j['sampled']:>10.4f}")

    noise_levels = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    success_probs = []
    print("\nNoise-strength scan at optimal iteration:")
    print(f" {'p2':>8} {'success':>10}")
    print(" " + "-" * 20)
    for p2 in noise_levels:
        cfg = NoiseConfig(p1=p2 / 10 if p2 > 0 else 0.0, p2=p2, p_meas=min(0.02, p2 / 2) if p2 > 0 else 0.0, enabled=(p2 > 0))
        counts = run_counts(n_qubits, targets, k_opt, shots, noise_cfg=cfg if p2 > 0 else None)
        p = success_probability(counts, targets, n_qubits, shots)
        success_probs.append(p)
        print(f" {p2:>8.3f} {p:>10.4f}")

    plot_amplitude_tracking(n_qubits, target, "week4_grover_amplitude_tracking.png")
    plot_iteration_sweep(
        ideal_rows,
        noisy_rows,
        title="Grover iteration sweep: ideal vs noisy (single target, n=4)",
        out_file="week4_grover_ideal_vs_noisy.png",
    )
    plot_noise_strength_scan(noise_levels, success_probs, "week4_grover_noise_scan.png")


def main() -> None:
    run_single_and_multi_target_experiments()
    run_capstone_oriented_noise_sweeps()

    print_header("WEEK 4 TAKEAWAYS")
    print("  1. Single-target and multi-target Grover searches both work in the ideal simulator.")
    print("  2. Success probability peaks near the usual optimal iteration count.")
    print("  3. Under a simple depolarizing + readout noise model, the success peak degrades.")
    print("  4. This directly motivates the capstone direction: Noise-aware Grover.")
    print("  5. A natural Week 6 goal is to quantify degradation versus both iteration count and noise strength.")


if __name__ == "__main__":
    main()