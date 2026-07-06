"""
Week 6 Capstone — Noise-aware Grover
====================================
Implements a polished capstone study of Grover's algorithm under a simple noise model.

Features
--------
1. Grover oracle and diffuser built from first principles
2. Single-target and multi-target search
3. Exact ideal target probabilities from statevector simulation
4. Shot-based ideal and noisy simulations using Qiskit Aer
5. Iteration sweeps showing success-probability degradation
6. Noise-strength sweeps at the ideal and experimentally best iterations
7. Circuit-depth / CX-count tracking to connect performance with noise accumulation
8. CSV export of quantitative results and PNG plots for the final report/demo

"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
import csv
import math
import os

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class NoiseConfig:
    p1: float = 0.001
    p2: float = 0.01
    p_meas: float = 0.01
    enabled: bool = True


@dataclass
class ExperimentSpec:
    name: str
    n_qubits: int
    targets: List[int]
    shots: int = 4096


# ─────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────

def bitstring_of(x: int, n_qubits: int) -> str:
    return format(x, f"0{n_qubits}b")


def validate_targets(n_qubits: int, targets: List[int]) -> None:
    N = 2 ** n_qubits
    if not targets:
        raise ValueError("targets must be non-empty")
    if len(set(targets)) != len(targets):
        raise ValueError("targets must be unique")
    bad = [t for t in targets if t < 0 or t >= N]
    if bad:
        raise ValueError(f"targets out of range for {n_qubits} qubits: {bad}")


def ensure_output_dir() -> None:
    os.makedirs("output", exist_ok=True)


def save_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────
# Grover theory
# ─────────────────────────────────────────────

def grover_angle(n_qubits: int, n_targets: int) -> float:
    N = 2 ** n_qubits
    return np.arcsin(np.sqrt(n_targets / N))


def theoretical_success_probability(n_qubits: int, n_targets: int, k: int) -> float:
    theta = grover_angle(n_qubits, n_targets)
    return float(np.sin((2 * k + 1) * theta) ** 2)


def optimal_iterations(n_qubits: int, n_targets: int = 1) -> int:
    theta = grover_angle(n_qubits, n_targets)
    return max(1, math.floor(np.pi / (4 * theta)))


# ─────────────────────────────────────────────
# Core Grover circuit construction
# ─────────────────────────────────────────────

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


def build_grover_circuit(
    n_qubits: int,
    targets: List[int],
    n_iterations: int,
    measure: bool = True,
    add_barriers: bool = False,
) -> QuantumCircuit:
    validate_targets(n_qubits, targets)

    qr = QuantumRegister(n_qubits, name="q")
    if measure:
        cr = ClassicalRegister(n_qubits, name="c")
        qc = QuantumCircuit(qr, cr)
    else:
        qc = QuantumCircuit(qr)

    qc.h(range(n_qubits))
    if add_barriers:
        qc.barrier(label="Init |s>")

    oracle = build_oracle(n_qubits, targets)
    diffuser = build_diffuser(n_qubits)

    for k in range(n_iterations):
        qc.append(oracle, range(n_qubits))
        qc.append(diffuser, range(n_qubits))
        if add_barriers:
            qc.barrier(label=f"Iter {k+1}")

    if measure:
        qc.measure(qr, range(n_qubits))

    return qc


# ─────────────────────────────────────────────
# Ideal and noisy simulators
# ─────────────────────────────────────────────

def build_simple_noise_model(cfg: NoiseConfig) -> NoiseModel:
    noise_model = NoiseModel()
    if not cfg.enabled:
        return noise_model

    err1 = depolarizing_error(cfg.p1, 1)
    err2 = depolarizing_error(cfg.p2, 2)
    readout = ReadoutError([[1 - cfg.p_meas, cfg.p_meas], [cfg.p_meas, 1 - cfg.p_meas]])

    for gate in ["x", "h", "z", "rz", "sx"]:
        try:
            noise_model.add_all_qubit_quantum_error(err1, gate)
        except Exception:
            pass

    noise_model.add_all_qubit_quantum_error(err2, ["cx"])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def run_counts(
    n_qubits: int,
    targets: List[int],
    n_iterations: int,
    shots: int,
    noise_cfg: Optional[NoiseConfig] = None,
) -> Dict[str, int]:
    qc = build_grover_circuit(n_qubits, targets, n_iterations, measure=True, add_barriers=False)
    sim = AerSimulator(noise_model=build_simple_noise_model(noise_cfg)) if noise_cfg else AerSimulator()
    tqc = transpile(qc, sim, optimization_level=1)
    result = sim.run(tqc, shots=shots).result()
    return result.get_counts(tqc)


def success_probability(counts: Dict[str, int], targets: List[int], n_qubits: int, shots: int) -> float:
    target_states = {bitstring_of(t, n_qubits) for t in targets}
    hits = sum(counts.get(bs, 0) for bs in target_states)
    return hits / shots


def exact_target_probability(n_qubits: int, targets: List[int], n_iterations: int) -> float:
    qc = build_grover_circuit(n_qubits, targets, n_iterations, measure=False, add_barriers=False)
    sv = Statevector.from_instruction(qc.decompose().decompose())
    probs = sv.probabilities_dict()
    target_states = {bitstring_of(t, n_qubits) for t in targets}
    return float(sum(probs.get(bs, 0.0) for bs in target_states))


# ─────────────────────────────────────────────
# Circuit metrics
# ─────────────────────────────────────────────

def circuit_metrics(n_qubits: int, targets: List[int], n_iterations: int) -> Dict[str, int]:
    qc = build_grover_circuit(n_qubits, targets, n_iterations, measure=True, add_barriers=False)
    tqc = transpile(qc.decompose().decompose(), basis_gates=["x", "h", "z", "rz", "sx", "cx"], optimization_level=1)
    ops = tqc.count_ops()
    return {
        "depth": int(tqc.depth()),
        "width": int(tqc.num_qubits),
        "cx_count": int(ops.get("cx", 0)),
        "measure_count": int(ops.get("measure", 0)),
        "one_qubit_count": int(sum(v for k, v in ops.items() if k not in {"cx", "measure", "barrier"})),
    }


# ─────────────────────────────────────────────
# Sweeps and analysis
# ─────────────────────────────────────────────

def iteration_sweep(
    spec: ExperimentSpec,
    max_iters: int,
    noise_cfg: Optional[NoiseConfig] = None,
) -> List[Dict]:
    rows = []
    for k in range(max_iters + 1):
        theory = theoretical_success_probability(spec.n_qubits, len(spec.targets), k)
        exact = exact_target_probability(spec.n_qubits, spec.targets, k)
        counts = run_counts(spec.n_qubits, spec.targets, k, spec.shots, noise_cfg=noise_cfg)
        sampled = success_probability(counts, spec.targets, spec.n_qubits, spec.shots)
        metrics = circuit_metrics(spec.n_qubits, spec.targets, k)
        rows.append({
            "experiment": spec.name,
            "n_qubits": spec.n_qubits,
            "n_targets": len(spec.targets),
            "targets": ",".join(map(str, spec.targets)),
            "iteration": k,
            "theory_success": theory,
            "exact_success": exact,
            "sampled_success": sampled,
            "depth": metrics["depth"],
            "cx_count": metrics["cx_count"],
            "one_qubit_count": metrics["one_qubit_count"],
        })
    return rows


def find_best_iteration(rows: List[Dict], key: str = "sampled_success") -> Tuple[int, float]:
    best = max(rows, key=lambda r: r[key])
    return int(best["iteration"]), float(best[key])


def noise_strength_sweep(
    spec: ExperimentSpec,
    noise_levels: List[float],
    fixed_iteration: int,
) -> List[Dict]:
    rows = []
    for p2 in noise_levels:
        cfg = NoiseConfig(
            p1=0.1 * p2,
            p2=p2,
            p_meas=min(0.02, 0.5 * p2),
            enabled=(p2 > 0),
        )
        counts = run_counts(spec.n_qubits, spec.targets, fixed_iteration, spec.shots, noise_cfg=cfg if p2 > 0 else None)
        sampled = success_probability(counts, spec.targets, spec.n_qubits, spec.shots)
        rows.append({
            "experiment": spec.name,
            "iteration": fixed_iteration,
            "p1": cfg.p1,
            "p2": cfg.p2,
            "p_meas": cfg.p_meas,
            "success_probability": sampled,
        })
    return rows


def summarize_degradation(ideal_rows: List[Dict], noisy_rows: List[Dict]) -> List[Dict]:
    summary = []
    for i_row, n_row in zip(ideal_rows, noisy_rows):
        summary.append({
            "experiment": i_row["experiment"],
            "iteration": i_row["iteration"],
            "ideal_sampled": i_row["sampled_success"],
            "noisy_sampled": n_row["sampled_success"],
            "absolute_drop": i_row["sampled_success"] - n_row["sampled_success"],
            "relative_drop_percent": 100 * (i_row["sampled_success"] - n_row["sampled_success"]) / max(i_row["sampled_success"], 1e-12),
        })
    return summary


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_iteration_comparison(ideal_rows: List[Dict], noisy_rows: List[Dict], spec: ExperimentSpec, out_file: str) -> None:
    xs = [r["iteration"] for r in ideal_rows]
    theory = [r["theory_success"] for r in ideal_rows]
    ideal = [r["sampled_success"] for r in ideal_rows]
    noisy = [r["sampled_success"] for r in noisy_rows]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(xs, theory, "--", color="#111827", linewidth=2, label="Theory")
    ax.plot(xs, ideal, "o-", color="#2563eb", linewidth=2, label="Ideal sampled")
    ax.plot(xs, noisy, "s-", color="#dc2626", linewidth=2, label="Noisy sampled")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Success probability")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.set_title(f"{spec.name}: ideal vs noisy Grover success")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_degradation(ideal_rows: List[Dict], noisy_rows: List[Dict], spec: ExperimentSpec, out_file: str) -> None:
    xs = [r["iteration"] for r in ideal_rows]
    drops = [i["sampled_success"] - n["sampled_success"] for i, n in zip(ideal_rows, noisy_rows)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(xs, drops, color="#7c3aed")
    ax.set_xlabel("Grover iterations")
    ax.set_ylabel("Absolute success-probability drop")
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(f"{spec.name}: noisy degradation relative to ideal")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_noise_scan(rows_ideal_k: List[Dict], rows_best_noisy_k: List[Dict], spec: ExperimentSpec, out_file: str) -> None:
    xs = [r["p2"] for r in rows_ideal_k]
    ys_ideal_k = [r["success_probability"] for r in rows_ideal_k]
    ys_best_noisy_k = [r["success_probability"] for r in rows_best_noisy_k]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs, ys_ideal_k, "o-", color="#2563eb", linewidth=2, label="At ideal k_opt")
    ax.plot(xs, ys_best_noisy_k, "s-", color="#dc2626", linewidth=2, label="At best noisy k")
    ax.set_xlabel("Two-qubit depolarizing probability p2")
    ax.set_ylabel("Success probability")
    ax.grid(alpha=0.25)
    ax.set_title(f"{spec.name}: noise-strength sweep")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_depth_vs_success(noisy_rows: List[Dict], spec: ExperimentSpec, out_file: str) -> None:
    depths = [r["depth"] for r in noisy_rows]
    probs = [r["sampled_success"] for r in noisy_rows]
    ks = [r["iteration"] for r in noisy_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    sc = ax.scatter(depths, probs, c=ks, cmap="viridis", s=80)
    ax.set_xlabel("Circuit depth")
    ax.set_ylabel("Noisy success probability")
    ax.grid(alpha=0.25)
    ax.set_title(f"{spec.name}: depth vs noisy success")
    cbar = plt.colorbar(sc)
    cbar.set_label("Grover iterations")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Polished demo runner
# ─────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 86)
    print(f" {title}")
    print("=" * 86)


def print_top_counts(counts: Dict[str, int], top_n: int = 6) -> None:
    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    for bs, c in ranked:
        print(f"  {bs}: {c}")


def polished_demo(spec: ExperimentSpec, noisy_cfg: NoiseConfig, max_iters: int) -> None:
    print_header(f"CAPSTONE DEMO — {spec.name}")
    k_opt = optimal_iterations(spec.n_qubits, len(spec.targets))
    print(f"Search space N = {2 ** spec.n_qubits}")
    print(f"Targets = {spec.targets} -> {[bitstring_of(t, spec.n_qubits) for t in spec.targets]}")
    print(f"Theoretical ideal iteration estimate k_opt = {k_opt}")

    ideal_rows = iteration_sweep(spec, max_iters=max_iters, noise_cfg=None)
    noisy_rows = iteration_sweep(spec, max_iters=max_iters, noise_cfg=noisy_cfg)
    degradation_rows = summarize_degradation(ideal_rows, noisy_rows)

    best_ideal_k, best_ideal_p = find_best_iteration(ideal_rows)
    best_noisy_k, best_noisy_p = find_best_iteration(noisy_rows)

    print(f"Best ideal sampled success:  k={best_ideal_k}, p={best_ideal_p:.4f}")
    print(f"Best noisy sampled success:  k={best_noisy_k}, p={best_noisy_p:.4f}")
    print(f"Drop at best ideal k: {next(r['absolute_drop'] for r in degradation_rows if r['iteration'] == best_ideal_k):.4f}")

    print("\nIteration-by-iteration summary:")
    print(f"{'k':>3} {'theory':>10} {'ideal':>10} {'noisy':>10} {'drop':>10} {'depth':>8} {'cx':>6}")
    print("-" * 66)
    for i_row, n_row, d_row in zip(ideal_rows, noisy_rows, degradation_rows):
        print(
            f"{i_row['iteration']:>3} {i_row['theory_success']:>10.4f} {i_row['sampled_success']:>10.4f} "
            f"{n_row['sampled_success']:>10.4f} {d_row['absolute_drop']:>10.4f} {n_row['depth']:>8} {n_row['cx_count']:>6}"
        )

    counts_ideal = run_counts(spec.n_qubits, spec.targets, best_ideal_k, spec.shots, noise_cfg=None)
    counts_noisy = run_counts(spec.n_qubits, spec.targets, best_noisy_k, spec.shots, noise_cfg=noisy_cfg)
    print("\nTop ideal counts at best ideal k:")
    print_top_counts(counts_ideal)
    print("Top noisy counts at best noisy k:")
    print_top_counts(counts_noisy)

    save_csv(f"output/{spec.name.lower().replace(' ', '_')}_ideal_rows.csv", ideal_rows)
    save_csv(f"output/{spec.name.lower().replace(' ', '_')}_noisy_rows.csv", noisy_rows)
    save_csv(f"output/{spec.name.lower().replace(' ', '_')}_degradation.csv", degradation_rows)

    plot_iteration_comparison(ideal_rows, noisy_rows, spec, f"output/{spec.name.lower().replace(' ', '_')}_ideal_vs_noisy.png")
    plot_degradation(ideal_rows, noisy_rows, spec, f"output/{spec.name.lower().replace(' ', '_')}_degradation.png")
    plot_depth_vs_success(noisy_rows, spec, f"output/{spec.name.lower().replace(' ', '_')}_depth_vs_success.png")

    noise_levels = [0.0, 0.002, 0.005, 0.01, 0.02, 0.04]
    rows_ideal_k = noise_strength_sweep(spec, noise_levels, fixed_iteration=k_opt)
    rows_best_noisy_k = noise_strength_sweep(spec, noise_levels, fixed_iteration=best_noisy_k)
    save_csv(f"output/{spec.name.lower().replace(' ', '_')}_noise_scan_ideal_k.csv", rows_ideal_k)
    save_csv(f"output/{spec.name.lower().replace(' ', '_')}_noise_scan_best_noisy_k.csv", rows_best_noisy_k)
    plot_noise_scan(rows_ideal_k, rows_best_noisy_k, spec, f"output/{spec.name.lower().replace(' ', '_')}_noise_scan.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main() -> None:
    ensure_output_dir()

    noisy_cfg = NoiseConfig(p1=0.001, p2=0.01, p_meas=0.01, enabled=True)

    experiments = [
        ExperimentSpec(name="Single Target n4", n_qubits=4, targets=[11], shots=4096),
        ExperimentSpec(name="Multi Target n4", n_qubits=4, targets=[3, 11], shots=4096),
        ExperimentSpec(name="Single Target n5", n_qubits=5, targets=[6], shots=4096),
    ]

    print_header("WEEK 6 CAPSTONE — NOISE-AWARE GROVER")
    print("Simple noise model: 1q depolarizing + 2q depolarizing + readout error")
    print(f"Base noisy configuration: p1={noisy_cfg.p1}, p2={noisy_cfg.p2}, p_meas={noisy_cfg.p_meas}")

    for spec in experiments:
        k_opt = optimal_iterations(spec.n_qubits, len(spec.targets))
        max_iters = max(2 * k_opt, k_opt + 3)
        polished_demo(spec, noisy_cfg=noisy_cfg, max_iters=max_iters)

    print_header("CAPSTONE OUTCOME")
    print("1. Ideal Grover success peaks near the standard theoretical iteration count.")
    print("2. Under the simple noise model, peak success probability is reduced and can shift in iteration index.")
    print("3. Success degrades as iteration count grows because circuit depth and CX count increase.")
    print("4. Increasing noise strength further suppresses success probability, illustrating the central capstone result.")
    print("5. The generated CSVs and PNGs provide a polished demonstration package for the final report.")


if __name__ == "__main__":
    main()