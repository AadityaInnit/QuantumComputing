"""
Grover's Search Algorithm
=========================
Searches an unsorted database of N=2^n items for a marked target
in O(√N) queries — a quadratic speedup over classical O(N) search.

Two core components:
  1. Oracle (Uf): Marks the target state by flipping its phase: |x⟩ → -|x⟩
  2. Diffusion operator (Grover diffuser): Amplifies the marked state's amplitude

After ≈ (π/4)√N iterations, the target is measured with high probability.

Skills demonstrated:
  - n-qubit oracle construction (phase kickback via multi-controlled-Z)
  - Grover diffusion operator (inversion about average)
  - Optimal iteration count calculation
  - Amplitude amplification intuition via probability plots
  - Multi-target search extension
  - Scalability analysis
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.circuit.library import GroverOperator
import numpy as np
import matplotlib.pyplot as plt
import math


# ─────────────────────────────────────────────
# Oracle Construction
# ─────────────────────────────────────────────

def build_oracle(n_qubits: int, targets: list) -> QuantumCircuit:
    """
    Build a phase-flip oracle that marks one or more target states.
    Implements: Uf|x⟩ = -|x⟩ if x ∈ targets, else |x⟩

    Strategy: For each target bitstring, flip qubits that are '0',
    apply an n-qubit controlled-Z (via H + MCX + H), then unflip.

    Args:
        n_qubits: Number of search qubits.
        targets:  List of integer targets to mark (e.g. [6] or [3, 7]).

    Returns:
        Oracle as a QuantumCircuit gate.
    """
    qc = QuantumCircuit(n_qubits, name="Oracle")

    for target in targets:
        # Convert target integer to n-bit binary string (big-endian)
        bitstring = format(target, f"0{n_qubits}b")

        # Flip qubits where bitstring has '0' (to convert target to all-|1⟩)
        for i, bit in enumerate(reversed(bitstring)):
            if bit == "0":
                qc.x(i)

        # Multi-controlled-Z: phase flip only when all qubits are |1⟩
        # Implemented as H · MCX · H on the last qubit
        qc.h(n_qubits - 1)
        qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
        qc.h(n_qubits - 1)

        # Unflip to restore basis
        for i, bit in enumerate(reversed(bitstring)):
            if bit == "0":
                qc.x(i)

    return qc


# ─────────────────────────────────────────────
# Diffusion Operator
# ─────────────────────────────────────────────

def build_diffuser(n_qubits: int) -> QuantumCircuit:
    """
    Grover diffusion operator: inversion about the average.
    D = 2|s⟩⟨s| - I  where |s⟩ = H^⊗n|0⟩^⊗n

    Circuit: H^⊗n · (2|0⟩⟨0| - I) · H^⊗n
    The inner reflection is implemented as:
        X^⊗n · H · MCX · H · X^⊗n

    Args:
        n_qubits: Number of qubits.

    Returns:
        Diffuser as a QuantumCircuit.
    """
    qc = QuantumCircuit(n_qubits, name="Diffuser")

    qc.h(range(n_qubits))   # rotate to computational basis
    qc.x(range(n_qubits))   # flip all so |0…0⟩ becomes |1…1⟩

    # Phase flip on |0…0⟩ ↔ controlled-Z on all-|1⟩
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))   # unflip
    qc.h(range(n_qubits))   # rotate back to superposition basis
    return qc


# ─────────────────────────────────────────────
# Full Grover Circuit
# ─────────────────────────────────────────────

def optimal_iterations(n_qubits: int, n_targets: int = 1) -> int:
    """
    Optimal number of Grover iterations:
        k ≈ (π/4) * sqrt(N / M)
    where N = 2^n (search space), M = number of targets.
    """
    N = 2 ** n_qubits
    return max(1, round((np.pi / 4) * np.sqrt(N / n_targets)))


def build_grover_circuit(n_qubits: int,
                         targets: list,
                         n_iterations: int = None) -> QuantumCircuit:
    """
    Build the full Grover search circuit.

    Args:
        n_qubits:     Number of search qubits (search space = 2^n).
        targets:      List of target integers to search for.
        n_iterations: Grover iterations (auto-computed if None).

    Returns:
        Full Grover circuit with measurement.
    """
    if n_iterations is None:
        n_iterations = optimal_iterations(n_qubits, len(targets))

    qr = QuantumRegister(n_qubits, name="q")
    cr = ClassicalRegister(n_qubits, name="c")
    qc = QuantumCircuit(qr, cr)

    # Step 1: Uniform superposition
    qc.h(range(n_qubits))
    qc.barrier(label=f"Init |s⟩")

    # Step 2: Repeat oracle + diffuser
    oracle   = build_oracle(n_qubits, targets)
    diffuser = build_diffuser(n_qubits)

    for iteration in range(n_iterations):
        qc.append(oracle,   range(n_qubits))
        qc.append(diffuser, range(n_qubits))
        qc.barrier(label=f"Iter {iteration+1}")

    # Step 3: Measure
    qc.measure(qr, cr)
    return qc


# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

def run_grover(n_qubits: int,
               targets: list,
               n_iterations: int = None,
               shots: int = 4096) -> dict:
    """
    Run Grover's algorithm and return measurement counts.

    Args:
        n_qubits:     Number of qubits.
        targets:      Target integers.
        n_iterations: Grover iterations (auto if None).
        shots:        Simulation shots.

    Returns:
        counts dict mapping bitstring → frequency.
    """
    qc = build_grover_circuit(n_qubits, targets, n_iterations)
    # Decompose custom-named subcircuits (Oracle, Diffuser) into
    # native gates that AerSimulator understands. Two passes handle
    # nested subcircuit wrapping from QuantumCircuit.append().
    qc_decomposed = qc.decompose().decompose()
    sim = AerSimulator()
    result = sim.run(qc_decomposed, shots=shots).result()
    return result.get_counts(qc_decomposed)


def success_probability(counts: dict, targets: list, n_qubits: int, shots: int) -> float:
    """Fraction of measurements that hit any target state."""
    target_bitstrings = {format(t, f"0{n_qubits}b") for t in targets}
    hits = sum(counts.get(bs, 0) for bs in target_bitstrings)
    return hits / shots


# ─────────────────────────────────────────────
# Amplitude Tracking (no measurement)
# ─────────────────────────────────────────────

def track_amplitudes(n_qubits: int, target: int, max_iters: int = None) -> dict:
    """
    Track the probability of finding the target state after each
    Grover iteration to visualise amplitude amplification.

    Args:
        n_qubits:  Number of qubits.
        target:    Single target integer.
        max_iters: How many iterations to track (default: 2× optimal).

    Returns:
        Dict {'target': [...], 'non_target': [...]} probabilities per iteration.
    """
    from qiskit.quantum_info import Statevector

    if max_iters is None:
        max_iters = 2 * optimal_iterations(n_qubits)

    oracle   = build_oracle(n_qubits, [target])
    diffuser = build_diffuser(n_qubits)
    target_bs = format(target, f"0{n_qubits}b")

    probs_target     = []
    probs_non_target = []

    # Initial state: uniform superposition
    init = QuantumCircuit(n_qubits)
    init.h(range(n_qubits))
    sv = Statevector.from_instruction(init)

    for it in range(max_iters + 1):
        prob_dict = sv.probabilities_dict()
        p_target = prob_dict.get(target_bs, 0)
        probs_target.append(p_target)
        probs_non_target.append(1 - p_target)

        if it < max_iters:
            step = QuantumCircuit(n_qubits)
            step.append(oracle,   range(n_qubits))
            step.append(diffuser, range(n_qubits))
            # Decompose custom gates so Statevector.evolve() can process them
            sv = sv.evolve(step.decompose().decompose())

    return {"target": probs_target, "non_target": probs_non_target}


# ─────────────────────────────────────────────
# Scalability Analysis
# ─────────────────────────────────────────────

def classical_vs_grover(max_qubits: int = 10) -> dict:
    """
    Compare expected classical queries vs Grover queries for varying n.

    Classical: N/2 on average
    Grover:    (π/4)√N
    """
    data = {"n": [], "N": [], "classical": [], "grover": []}
    for n in range(1, max_qubits + 1):
        N = 2 ** n
        data["n"].append(n)
        data["N"].append(N)
        data["classical"].append(N / 2)
        data["grover"].append((np.pi / 4) * np.sqrt(N))
    return data


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_amplitude_amplification(n_qubits: int = 4, target: int = 6) -> None:
    """Show probability of finding target vs iteration count."""
    k_opt = optimal_iterations(n_qubits)
    max_iters = max(2 * k_opt, 6)
    probs = track_amplitudes(n_qubits, target, max_iters)

    iters = list(range(max_iters + 1))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(iters, probs["target"],     color="#4A90D9", linewidth=2.5,
            marker="o", markersize=6, label=f"P(target={target})")
    ax.plot(iters, probs["non_target"], color="#BDC3C7", linewidth=1.5,
            linestyle="--", label="P(non-target)")
    ax.axvline(k_opt, color="#E74C3C", linestyle=":", linewidth=2,
               label=f"Optimal k={k_opt}")
    ax.axhline(1 / (2 ** n_qubits), color="#95A5A6", linestyle=":",
               linewidth=1, label=f"Initial prob = 1/{2**n_qubits}")
    ax.set_xlabel("Grover Iterations")
    ax.set_ylabel("Probability")
    ax.set_title(f"Amplitude Amplification — n={n_qubits} qubits, N={2**n_qubits} states",
                 fontsize=13, fontweight="bold")
    ax.legend(); ax.set_ylim(-0.05, 1.05); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("grover_amplitude_amplification.png", dpi=150)
    plt.show()
    print("Amplitude plot saved to grover_amplitude_amplification.png")


def plot_measurement_histogram(counts: dict, n_qubits: int,
                               targets: list, shots: int) -> None:
    """Plot measurement outcomes, highlighting target states."""
    N = 2 ** n_qubits
    target_bs = {format(t, f"0{n_qubits}b") for t in targets}
    all_states = {format(i, f"0{n_qubits}b"): counts.get(format(i, f"0{n_qubits}b"), 0)
                  for i in range(N)}
    colors = ["#E74C3C" if bs in target_bs else "#4A90D9"
              for bs in all_states]

    fig, ax = plt.subplots(figsize=(max(10, N // 2), 5))
    bars = ax.bar(all_states.keys(), all_states.values(),
                  color=colors, edgecolor="white", width=0.7)
    ax.set_xlabel("State (bitstring)")
    ax.set_ylabel("Counts")
    target_labels = ", ".join([f"|{bs}⟩ ({t})" for t, bs in
                               zip(targets, [format(t, f"0{n_qubits}b") for t in targets])])
    ax.set_title(f"Grover Search Results — Target: {target_labels}\n"
                 f"n={n_qubits} qubits, N={N} states, {shots} shots",
                 fontsize=12, fontweight="bold")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#E74C3C", label="Target state(s)"),
                        Patch(color="#4A90D9",  label="Non-target states")])
    plt.xticks(rotation=45 if N > 16 else 0, fontsize=8)
    plt.tight_layout()
    plt.savefig("grover_measurement.png", dpi=150)
    plt.show()
    print("Measurement histogram saved to grover_measurement.png")


def plot_speedup_comparison() -> None:
    """Log-scale comparison of classical vs Grover query complexity."""
    data = classical_vs_grover(max_qubits=14)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Linear scale
    ax = axes[0]
    ax.plot(data["n"], data["classical"], "o-", color="#E74C3C",
            label="Classical (N/2)", linewidth=2)
    ax.plot(data["n"], data["grover"],    "s-", color="#4A90D9",
            label="Grover (π√N/4)", linewidth=2)
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Expected queries")
    ax.set_title("Linear scale"); ax.legend(); ax.grid(alpha=0.3)

    # Log scale
    ax = axes[1]
    ax.semilogy(data["n"], data["classical"], "o-", color="#E74C3C",
                label="Classical (N/2)", linewidth=2)
    ax.semilogy(data["n"], data["grover"],    "s-", color="#4A90D9",
                label="Grover (π√N/4)", linewidth=2)
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Expected queries (log scale)")
    ax.set_title("Log scale — speedup widens exponentially"); ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("Grover's Algorithm vs Classical Search — Query Complexity",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("grover_speedup.png", dpi=150)
    plt.show()
    print("Speedup plot saved to grover_speedup.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # ── Configuration ──────────────────────────
    N_QUBITS   = 4          # search space: 2^4 = 16 states
    TARGET     = 11         # the "marked" state to find  (0–15)
    SHOTS      = 4096

    N          = 2 ** N_QUBITS
    K_OPT      = optimal_iterations(N_QUBITS)
    TARGET_BS  = format(TARGET, f"0{N_QUBITS}b")

    print("\n" + "=" * 65)
    print("  GROVER'S SEARCH ALGORITHM")
    print(f"  Search space : 2^{N_QUBITS} = {N} states")
    print(f"  Target       : {TARGET} (|{TARGET_BS}⟩)")
    print(f"  Classical expected queries : {N//2}")
    print(f"  Grover optimal iterations  : {K_OPT}  (≈ π√N/4 = {np.pi/4*np.sqrt(N):.2f})")
    print(f"  Speedup factor             : {(N//2)/K_OPT:.1f}×")
    print("=" * 65)

    # Circuit diagram (decomposed components)
    print("\nOracle circuit:")
    print(build_oracle(N_QUBITS, [TARGET]).decompose().draw(output="text"))
    print("\nDiffuser circuit:")
    print(build_diffuser(N_QUBITS).decompose().draw(output="text"))

    # Full Grover circuit
    qc = build_grover_circuit(N_QUBITS, [TARGET])
    print(f"\nFull Grover circuit ({K_OPT} iterations):")
    print(qc.draw(output="text", fold=120))

    # ── Run single-target search ───────────────
    print(f"\nRunning Grover search (k={K_OPT} iterations, {SHOTS} shots)...")
    counts = run_grover(N_QUBITS, [TARGET], shots=SHOTS)
    p_success = success_probability(counts, [TARGET], N_QUBITS, SHOTS)
    print(f"  Target '{TARGET_BS}' measured : {counts.get(TARGET_BS, 0)} times")
    print(f"  Success probability : {p_success*100:.1f}%  (classical: {100/N:.1f}%)")

    # ── Multi-target search ────────────────────
    MULTI_TARGETS = [3, 11]
    K_MULTI = optimal_iterations(N_QUBITS, len(MULTI_TARGETS))
    print(f"\nMulti-target search: {MULTI_TARGETS}, k={K_MULTI} iterations")
    counts_multi = run_grover(N_QUBITS, MULTI_TARGETS, n_iterations=K_MULTI, shots=SHOTS)
    p_multi = success_probability(counts_multi, MULTI_TARGETS, N_QUBITS, SHOTS)
    print(f"  Combined success probability : {p_multi*100:.1f}%")

    # ── Iteration sweep ────────────────────────
    print(f"\nProbability of finding target vs iteration count:")
    print(f"  {'Iteration':>10}  {'P(target)':>12}  {'Bar'}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*20}")
    probs = track_amplitudes(N_QUBITS, TARGET, max_iters=2*K_OPT)
    for it, p in enumerate(probs["target"]):
        bar = "█" * int(p * 40)
        marker = " ← optimal" if it == K_OPT else ""
        print(f"  {it:>10}  {p:>11.4f}  {bar}{marker}")

    # ── Plots ──────────────────────────────────
    print()
    plot_amplitude_amplification(N_QUBITS, TARGET)
    plot_measurement_histogram(counts, N_QUBITS, [TARGET], SHOTS)
    plot_speedup_comparison()

    print("\n  KEY INSIGHT:")
    print(f"  Classical search needs O(N/2)={N//2} queries on average.")
    print(f"  Grover finds the answer in O(√N)≈{K_OPT} quantum queries.")
    print(f"  This √N speedup is provably optimal for unstructured search.")
