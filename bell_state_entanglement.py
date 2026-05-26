"""
Bell State Entanglement
=======================
Creates the four maximally entangled Bell states using Hadamard
and CNOT gates. Demonstrates quantum correlations: once measured,
both qubits are always perfectly correlated (or anti-correlated),
regardless of any classical explanation — the essence of entanglement.

Bell states:
  |Φ+⟩ = (|00⟩ + |11⟩) / √2   ← most common demo state
  |Φ-⟩ = (|00⟩ - |11⟩) / √2
  |Ψ+⟩ = (|01⟩ + |10⟩) / √2
  |Ψ-⟩ = (|01⟩ - |10⟩) / √2

Skills demonstrated:
  - Two-qubit circuits
  - H + CNOT entanglement protocol
  - X and Z gates for Bell state selection
  - Statevector simulation (exact amplitudes)
  - Measurement correlation analysis
  - Plotting joint probability distributions
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import matplotlib.pyplot as plt
import numpy as np


# ─────────────────────────────────────────────
# Bell State Circuit Builders
# ─────────────────────────────────────────────

def build_bell_state(state: str = "phi_plus") -> QuantumCircuit:
    """
    Build a Bell state preparation circuit.

    Args:
        state: One of 'phi_plus', 'phi_minus', 'psi_plus', 'psi_minus'.

    Returns:
        QuantumCircuit that prepares the selected Bell state.
    """
    qr = QuantumRegister(2, name="q")
    cr = ClassicalRegister(2, name="c")
    qc = QuantumCircuit(qr, cr)

    # Step 1: Apply X gate to q[0] to select Ψ states (|01⟩/|10⟩ basis)
    if state in ("psi_plus", "psi_minus"):
        qc.x(qr[0])

    # Step 2: Hadamard on q[1] — creates superposition on control qubit
    qc.h(qr[1])

    # Step 3: CNOT — entangles q[1] (control) with q[0] (target)
    qc.cx(qr[1], qr[0])

    # Step 4: Apply Z to q[1] to select minus-phase states
    if state in ("phi_minus", "psi_minus"):
        qc.z(qr[1])

    # Measurement
    qc.measure(qr, cr)
    return qc


def build_bell_state_no_measure(state: str = "phi_plus") -> QuantumCircuit:
    """Bell state circuit without measurement (for statevector inspection)."""
    qr = QuantumRegister(2, name="q")
    qc = QuantumCircuit(qr)

    if state in ("psi_plus", "psi_minus"):
        qc.x(qr[0])
    qc.h(qr[1])
    qc.cx(qr[1], qr[0])
    if state in ("phi_minus", "psi_minus"):
        qc.z(qr[1])
    return qc


# ─────────────────────────────────────────────
# Statevector Analysis
# ─────────────────────────────────────────────

def analyse_statevector(state: str) -> None:
    """Print the exact quantum amplitudes for a Bell state."""
    qc = build_bell_state_no_measure(state)
    sv = Statevector.from_instruction(qc)
    data = sv.data  # complex amplitudes

    BELL_LABELS = {
        "phi_plus" : "|Φ+⟩ = (|00⟩ + |11⟩)/√2",
        "phi_minus": "|Φ-⟩ = (|00⟩ − |11⟩)/√2",
        "psi_plus" : "|Ψ+⟩ = (|01⟩ + |10⟩)/√2",
        "psi_minus": "|Ψ-⟩ = (|01⟩ − |10⟩)/√2",
    }
    basis = ["|00⟩", "|01⟩", "|10⟩", "|11⟩"]

    print(f"\n  State : {BELL_LABELS[state]}")
    print(f"  {'Basis':>6}  {'Amplitude':>20}  {'Probability':>12}")
    print(f"  {'─'*6}  {'─'*20}  {'─'*12}")
    for label, amp in zip(basis, data):
        prob = abs(amp) ** 2
        amp_str = f"{amp.real:+.4f}" if abs(amp.imag) < 1e-10 else f"{amp.real:+.4f}{amp.imag:+.4f}j"
        print(f"  {label:>6}  {amp_str:>20}  {prob:>11.4f}")


# ─────────────────────────────────────────────
# Measurement & Correlation Analysis
# ─────────────────────────────────────────────

def run_bell_experiment(state: str = "phi_plus", shots: int = 4096) -> dict:
    """
    Simulate measurement of a Bell state.

    Args:
        state: Bell state label.
        shots: Number of measurement shots.

    Returns:
        counts: Measurement outcome frequencies.
    """
    qc = build_bell_state(state)
    simulator = AerSimulator()
    job = simulator.run(qc, shots=shots)
    result = job.result()
    return result.get_counts(qc)


def analyse_correlations(counts: dict, state: str, shots: int) -> None:
    """Print correlation statistics from measurement counts."""
    correlated     = counts.get("00", 0) + counts.get("11", 0)   # same outcomes
    anti_correlated = counts.get("01", 0) + counts.get("10", 0)  # opposite outcomes

    print(f"\n  Correlation analysis ({state}, {shots} shots):")
    print(f"  {'Outcome':>8}  {'Count':>6}  {'Probability':>12}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*12}")
    for outcome in ["00", "01", "10", "11"]:
        cnt = counts.get(outcome, 0)
        prob = cnt / shots
        print(f"  {outcome:>8}  {cnt:>6}  {prob:>11.4f}")
    print()
    print(f"  Correlated outcomes     (00 + 11): {correlated/shots*100:.1f}%")
    print(f"  Anti-correlated outcomes (01 + 10): {anti_correlated/shots*100:.1f}%")


# ─────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────

def plot_all_bell_states(shots: int = 4096) -> None:
    """Run all four Bell states and plot their measurement histograms."""
    states = ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]
    labels = ["|Φ+⟩", "|Φ-⟩", "|Ψ+⟩", "|Ψ-⟩"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    colors = ["#4A90D9", "#E74C3C", "#2ECC71", "#F39C12"]

    for idx, (state, label) in enumerate(zip(states, labels)):
        counts = run_bell_experiment(state, shots=shots)
        all_outcomes = {"00": 0, "01": 0, "10": 0, "11": 0}
        all_outcomes.update(counts)

        ax = axes[idx]
        bars = ax.bar(all_outcomes.keys(), all_outcomes.values(),
                      color=colors[idx], edgecolor="white", width=0.5)
        ax.set_title(f"Bell State {label}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Measurement Outcome")
        ax.set_ylabel("Counts")
        ax.set_ylim(0, shots * 0.65)

        # Annotate bars
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 20,
                        f"{h/shots*100:.1f}%", ha="center", va="bottom", fontsize=10)

    fig.suptitle(f"Four Bell States — {shots} shots each", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("bell_states_all.png", dpi=150)
    plt.show()
    print("Plot saved to bell_states_all.png")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    SHOTS = 4096
    TARGET_STATE = "phi_plus"   # Change to explore other Bell states

    print("\n" + "=" * 55)
    print("  BELL STATE ENTANGLEMENT DEMO")
    print("=" * 55)

    # 1. Show circuit
    qc = build_bell_state(TARGET_STATE)
    print(f"\nCircuit for {TARGET_STATE}:")
    print(qc.draw(output="text"))

    # 2. Exact statevector analysis (no measurement noise)
    print("\nStatevector amplitudes (all four Bell states):")
    print("-" * 55)
    for state in ["phi_plus", "phi_minus", "psi_plus", "psi_minus"]:
        analyse_statevector(state)

    # 3. Measurement simulation for target state
    print("\n" + "-" * 55)
    print(f"Measurement results for {TARGET_STATE} ({SHOTS} shots):")
    counts = run_bell_experiment(TARGET_STATE, shots=SHOTS)
    analyse_correlations(counts, TARGET_STATE, shots=SHOTS)

    print("\nKey insight:")
    print("  For |Φ+⟩ and |Φ-⟩: outcomes are always 00 or 11.")
    print("  For |Ψ+⟩ and |Ψ-⟩: outcomes are always 01 or 10.")
    print("  This perfect correlation/anti-correlation persists")
    print("  no matter how far apart the qubits are — entanglement!")

    # 4. Plot all four Bell states
    print()
    plot_all_bell_states(shots=SHOTS)
