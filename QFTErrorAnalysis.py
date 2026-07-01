"""
Week 5 — QFT, Basic Phase Estimation, and Fidelity/Error Analysis
=================================================================
Implements:
1. Manual Quantum Fourier Transform (QFT) and inverse QFT
2. QFT validation against library implementation and identity checks
3. Basic Quantum Phase Estimation (QPE) circuits for simple unitaries/eigenstates
4. Fidelity and phase-estimation error analysis
5. A small Grover-refinement utility to keep the Noise-aware Grover capstone on track

Designed to satisfy Week 5 of the SoS PoA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.circuit.library import QFT
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError


# ─────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────

@dataclass
class QPEResult:
    label: str
    true_phase: float
    estimated_phase: float
    dominant_bitstring: str
    success_probability: float
    absolute_error: float


@dataclass
class NoiseConfig:
    p1: float = 0.001
    p2: float = 0.01
    p_meas: float = 0.01
    enabled: bool = False


# ─────────────────────────────────────────────
# QFT implementation
# ─────────────────────────────────────────────

def apply_qft_rotations(qc: QuantumCircuit, qubits: List[int]) -> None:
    n = len(qubits)
    if n == 0:
        return

    target = qubits[-1]
    qc.h(target)
    for j in range(n - 1):
        control = qubits[j]
        angle = np.pi / (2 ** (n - 1 - j))
        qc.cp(angle, control, target)
    apply_qft_rotations(qc, qubits[:-1])


def swap_register(qc: QuantumCircuit, qubits: List[int]) -> None:
    n = len(qubits)
    for i in range(n // 2):
        qc.swap(qubits[i], qubits[n - i - 1])


def build_manual_qft(n_qubits: int, do_swaps: bool = True, inverse: bool = False) -> QuantumCircuit:
    qc = QuantumCircuit(n_qubits, name="QFT†" if inverse else "QFT")

    if not inverse:
        apply_qft_rotations(qc, list(range(n_qubits)))
        if do_swaps:
            swap_register(qc, list(range(n_qubits)))
    else:
        if do_swaps:
            swap_register(qc, list(range(n_qubits)))
        qc_comp = build_manual_qft(n_qubits, do_swaps=False, inverse=False)
        qc = qc.compose(qc_comp.inverse())
        if do_swaps:
            qc = qc.compose(QuantumCircuit(n_qubits))
        qc.name = "QFT†"

    return qc


def qft_on_basis_state(n_qubits: int, basis_index: int, use_library: bool = False) -> Statevector:
    qc = QuantumCircuit(n_qubits)
    bitstring = format(basis_index, f"0{n_qubits}b")
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(i)

    qft_circ = QFT(n_qubits, do_swaps=True) if use_library else build_manual_qft(n_qubits, do_swaps=True)
    qc.append(qft_circ, range(n_qubits))
    return Statevector.from_instruction(qc.decompose())


def validate_qft_identity(n_qubits: int) -> float:
    qc = QuantumCircuit(n_qubits)
    qc.append(build_manual_qft(n_qubits), range(n_qubits))
    qc.append(build_manual_qft(n_qubits, inverse=True), range(n_qubits))
    sv_out = Statevector.from_instruction(qc.decompose())
    sv_in = Statevector.from_label("0" * n_qubits)
    return float(state_fidelity(sv_out, sv_in))


def compare_manual_vs_library_qft(n_qubits: int) -> float:
    manual = build_manual_qft(n_qubits, do_swaps=True)
    library = QFT(n_qubits, do_swaps=True)
    qc_manual = QuantumCircuit(n_qubits)
    qc_library = QuantumCircuit(n_qubits)
    qc_manual.append(manual, range(n_qubits))
    qc_library.append(library, range(n_qubits))
    sv_manual = Statevector.from_instruction(qc_manual.decompose())
    sv_library = Statevector.from_instruction(qc_library.decompose())
    return float(state_fidelity(sv_manual, sv_library))


# ─────────────────────────────────────────────
# QPE implementation
# ─────────────────────────────────────────────

def append_controlled_unitary_power(qc: QuantumCircuit, control: int, target: int, phase: float, power: int) -> None:
    qc.cp(2 * np.pi * phase * power, control, target)


def build_qpe_circuit(
    n_count: int,
    phase: float,
    eigenstate: str = "1",
    measure: bool = True,
    use_manual_iqft: bool = True,
) -> QuantumCircuit:
    total_qubits = n_count + 1
    qr = QuantumRegister(total_qubits, name="q")
    cr = ClassicalRegister(n_count, name="c") if measure else None
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    counting = list(range(n_count))
    target = n_count

    if eigenstate == "1":
        qc.x(target)
    elif eigenstate != "0":
        raise ValueError("Only eigenstates '0' and '1' are supported in this basic QPE demo")

    qc.h(counting)

    for j in range(n_count):
        power = 2 ** j
        append_controlled_unitary_power(qc, counting[j], target, phase, power)

    iqft = build_manual_qft(n_count, do_swaps=True, inverse=True) if use_manual_iqft else QFT(n_count, inverse=True, do_swaps=True)
    qc.append(iqft, counting)

    if measure:
        qc.measure(counting, range(n_count))

    return qc


def bitstring_to_phase(bitstring: str) -> float:
    return int(bitstring, 2) / (2 ** len(bitstring))


def run_qpe(
    n_count: int,
    phase: float,
    shots: int = 4096,
    noise_cfg: Optional[NoiseConfig] = None,
    label: str = "",
) -> QPEResult:
    qc = build_qpe_circuit(n_count=n_count, phase=phase, eigenstate="1", measure=True, use_manual_iqft=True)

    if noise_cfg is not None and noise_cfg.enabled:
        sim = AerSimulator(noise_model=build_simple_noise_model(noise_cfg))
    else:
        sim = AerSimulator()

    tqc = transpile(qc, sim, optimization_level=1)
    counts = sim.run(tqc, shots=shots).result().get_counts(tqc)
    dominant = max(counts, key=counts.get)
    est = bitstring_to_phase(dominant)
    success_prob = counts[dominant] / shots
    err = min(abs(est - phase), 1 - abs(est - phase))

    return QPEResult(
        label=label or f"phase={phase}",
        true_phase=phase,
        estimated_phase=est,
        dominant_bitstring=dominant,
        success_probability=success_prob,
        absolute_error=err,
    )


# ─────────────────────────────────────────────
# Fidelity / error analysis
# ─────────────────────────────────────────────

def qft_reconstruction_fidelity(n_qubits: int, basis_index: int) -> float:
    qc = QuantumCircuit(n_qubits)
    bitstring = format(basis_index, f"0{n_qubits}b")
    for i, bit in enumerate(reversed(bitstring)):
        if bit == "1":
            qc.x(i)

    original = Statevector.from_instruction(qc)
    qc.append(build_manual_qft(n_qubits), range(n_qubits))
    qc.append(build_manual_qft(n_qubits, inverse=True), range(n_qubits))
    recovered = Statevector.from_instruction(qc.decompose())
    return float(state_fidelity(original, recovered))


def qpe_statevector_overlap(n_count: int, phase: float) -> float:
    qc_manual = build_qpe_circuit(n_count=n_count, phase=phase, measure=False, use_manual_iqft=True)
    qc_lib = build_qpe_circuit(n_count=n_count, phase=phase, measure=False, use_manual_iqft=False)
    sv_manual = Statevector.from_instruction(qc_manual.decompose())
    sv_lib = Statevector.from_instruction(qc_lib.decompose())
    return float(state_fidelity(sv_manual, sv_lib))


def phase_error_table(phases: List[float], n_count: int, shots: int, noise_cfg: Optional[NoiseConfig] = None) -> List[QPEResult]:
    return [run_qpe(n_count=n_count, phase=phi, shots=shots, noise_cfg=noise_cfg, label=f"phi={phi}") for phi in phases]


# ─────────────────────────────────────────────
# Noise model
# ─────────────────────────────────────────────

def build_simple_noise_model(cfg: NoiseConfig) -> NoiseModel:
    noise_model = NoiseModel()
    if not cfg.enabled:
        return noise_model

    err1 = depolarizing_error(cfg.p1, 1)
    err2 = depolarizing_error(cfg.p2, 2)
    ro = ReadoutError([[1 - cfg.p_meas, cfg.p_meas], [cfg.p_meas, 1 - cfg.p_meas]])

    for gate in ["x", "h", "z", "sx", "rz", "cp", "swap"]:
        try:
            noise_model.add_all_qubit_quantum_error(err1, gate)
        except Exception:
            pass
    for gate in ["cx"]:
        noise_model.add_all_qubit_quantum_error(err2, gate)

    noise_model.add_all_qubit_readout_error(ro)
    return noise_model


# ─────────────────────────────────────────────
# Grover refinement helper
# ─────────────────────────────────────────────

def grover_theoretical_success(n_qubits: int, n_targets: int, k: int) -> float:
    theta = np.arcsin(np.sqrt(n_targets / (2 ** n_qubits)))
    return float(np.sin((2 * k + 1) * theta) ** 2)


def grover_optimal_iterations(n_qubits: int, n_targets: int = 1) -> int:
    return max(1, round((np.pi / 4) * np.sqrt((2 ** n_qubits) / n_targets)))


def grover_iteration_tradeoff_table(n_qubits: int, n_targets: int, max_extra: int = 3) -> List[Dict[str, float]]:
    k_opt = grover_optimal_iterations(n_qubits, n_targets)
    rows = []
    for k in range(max(0, k_opt - max_extra), k_opt + max_extra + 1):
        rows.append({
            "k": k,
            "theory_success": grover_theoretical_success(n_qubits, n_targets, k),
            "distance_from_opt": k - k_opt,
        })
    return rows


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_qpe_errors(results_ideal: List[QPEResult], results_noisy: List[QPEResult], out_file: str) -> None:
    labels = [r.label for r in results_ideal]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [r.absolute_error for r in results_ideal], width, label="Ideal", color="#2563eb")
    ax.bar(x + width / 2, [r.absolute_error for r in results_noisy], width, label="Noisy", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Absolute phase error")
    ax.set_title("QPE phase-estimation error: ideal vs noisy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_qpe_success(results_ideal: List[QPEResult], results_noisy: List[QPEResult], out_file: str) -> None:
    labels = [r.label for r in results_ideal]
    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, [r.success_probability for r in results_ideal], width, label="Ideal", color="#059669")
    ax.bar(x + width / 2, [r.success_probability for r in results_noisy], width, label="Noisy", color="#f59e0b")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Dominant-outcome probability")
    ax.set_ylim(0, 1.0)
    ax.set_title("QPE dominant-outcome probability: ideal vs noisy")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


def plot_grover_tradeoff(rows: List[Dict[str, float]], out_file: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r["k"] for r in rows], [r["theory_success"] for r in rows], "o-", color="#7c3aed", linewidth=2)
    ax.set_xlabel("Grover iterations k")
    ax.set_ylabel("Theoretical success probability")
    ax.set_title("Grover iteration trade-off near the optimum")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close()


# ─────────────────────────────────────────────
# Main experiment suite
# ─────────────────────────────────────────────

def print_header(title: str) -> None:
    print("\n" + "=" * 78)
    print(f" {title}")
    print("=" * 78)


def main() -> None:
    print_header("WEEK 5 — QFT VALIDATION")
    for n in [2, 3, 4]:
        fid_id = validate_qft_identity(n)
        fid_lib = compare_manual_vs_library_qft(n)
        rec_fid = qft_reconstruction_fidelity(n, basis_index=min(3, 2 ** n - 1))
        print(f"n={n}: F(QFT†QFT, I)={fid_id:.6f}, F(manual, library)={fid_lib:.6f}, reconstruction fidelity={rec_fid:.6f}")

    print_header("WEEK 5 — BASIC QPE")
    phases = [0.125, 0.25, 0.375, 0.625]
    n_count = 3
    shots = 4096
    noisy_cfg = NoiseConfig(p1=0.001, p2=0.01, p_meas=0.01, enabled=True)

    ideal_results = phase_error_table(phases, n_count=n_count, shots=shots, noise_cfg=None)
    noisy_results = phase_error_table(phases, n_count=n_count, shots=shots, noise_cfg=noisy_cfg)

    print(f"{'Phase':>8} {'Ideal est':>10} {'Ideal err':>10} {'Noisy est':>10} {'Noisy err':>10}")
    print("-" * 58)
    for rid, rnoisy in zip(ideal_results, noisy_results):
        print(f"{rid.true_phase:>8.3f} {rid.estimated_phase:>10.3f} {rid.absolute_error:>10.4f} {rnoisy.estimated_phase:>10.3f} {rnoisy.absolute_error:>10.4f}")

    overlap = qpe_statevector_overlap(n_count=3, phase=0.125)
    print(f"\nManual-iQFT vs library-iQFT state overlap (phase=0.125): {overlap:.6f}")

    print_header("WEEK 5 — GROVER REFINEMENT")
    rows = grover_iteration_tradeoff_table(n_qubits=5, n_targets=1, max_extra=3)
    print(f"{'k':>4} {'theory_success':>18} {'delta_from_opt':>16}")
    print("-" * 42)
    for r in rows:
        print(f"{r['k']:>4} {r['theory_success']:>18.6f} {r['distance_from_opt']:>16}")

    plot_qpe_errors(ideal_results, noisy_results, "week5_qpe_errors.png")
    plot_qpe_success(ideal_results, noisy_results, "week5_qpe_success.png")
    plot_grover_tradeoff(rows, "week5_grover_tradeoff.png")


if __name__ == "__main__":
    main()