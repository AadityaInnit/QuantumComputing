"""
Quantum Coin Flip
=================
Demonstrates superposition using the Hadamard gate.
A qubit in |0⟩ is placed in superposition and measured,
yielding 'Heads' or 'Tails' with equal ~50% probability.

Skills demonstrated:
  - Circuit construction (QuantumCircuit)
  - Hadamard gate (H gate)
  - Measurement and classical register
  - Simulation with AerSimulator
  - Statistical analysis over multiple trials
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt


def build_coin_flip_circuit() -> QuantumCircuit:
    """Build a single-qubit coin flip circuit."""
    qr = QuantumRegister(1, name="coin")
    cr = ClassicalRegister(1, name="result")
    qc = QuantumCircuit(qr, cr)

    # Place qubit in superposition: |0⟩ → (|0⟩ + |1⟩) / √2
    qc.h(qr[0])

    # Measure: collapses to |0⟩ or |1⟩ with equal probability
    qc.measure(qr, cr)
    return qc


def run_coin_flip(num_trials: int = 1000) -> dict:
    """
    Run the coin flip experiment.

    Args:
        num_trials: Number of shots (flips) to simulate.

    Returns:
        counts: Dict mapping '0' (Heads) and '1' (Tails) to their frequencies.
    """
    qc = build_coin_flip_circuit()
    simulator = AerSimulator()

    # Transpile is handled internally by AerSimulator.run()
    job = simulator.run(qc, shots=num_trials)
    result = job.result()
    counts = result.get_counts(qc)
    return counts


def display_results(counts: dict, num_trials: int) -> None:
    """Print a summary and show a histogram of outcomes."""
    heads = counts.get("0", 0)  # |0⟩ = Heads
    tails = counts.get("1", 0)  # |1⟩ = Tails

    print("=" * 40)
    print("  QUANTUM COIN FLIP RESULTS")
    print("=" * 40)
    print(f"  Total flips : {num_trials}")
    print(f"  Heads (|0⟩) : {heads}  ({heads/num_trials*100:.1f}%)")
    print(f"  Tails (|1⟩) : {tails}  ({tails/num_trials*100:.1f}%)")
    print("=" * 40)
    print("  Theoretical probability: 50% / 50%")
    print()

    # Rename keys for readability in histogram
    readable = {"Heads |0⟩": heads, "Tails |1⟩": tails}
    fig = plot_histogram(readable, title=f"Quantum Coin Flip — {num_trials} Trials")
    plt.tight_layout()
    plt.savefig("quantum_coin_flip_histogram.png", dpi=150)
    plt.show()
    print("Histogram saved to quantum_coin_flip_histogram.png")


if __name__ == "__main__":
    NUM_TRIALS = 1000

    # Show the circuit
    qc = build_coin_flip_circuit()
    print("\nCircuit diagram:")
    print(qc.draw(output="text"))
    print()

    # Run experiment
    counts = run_coin_flip(num_trials=NUM_TRIALS)
    display_results(counts, NUM_TRIALS)
