"""
Quantum Random Number Generator (QRNG)
=======================================
Uses quantum superposition to generate *true* random numbers —
unlike classical PRNGs, randomness here is fundamental to physics,
not algorithmic. Ideal as a cryptography primer.

Skills demonstrated:
  - Multi-qubit circuits
  - Parallel Hadamard gates for independent random bits
  - Binary-to-integer conversion
  - Generating random numbers in a bounded range [min_val, max_val]
  - Histogram analysis of uniformity
"""

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from collections import Counter
import matplotlib.pyplot as plt
import math


def build_qrng_circuit(num_bits: int) -> QuantumCircuit:
    """
    Build an n-qubit QRNG circuit.
    Each qubit is independently placed in superposition and measured,
    producing a uniformly random n-bit string.

    Args:
        num_bits: Number of random bits (determines range: 0 to 2^num_bits - 1).
    """
    qr = QuantumRegister(num_bits, name="q")
    cr = ClassicalRegister(num_bits, name="bits")
    qc = QuantumCircuit(qr, cr)

    # Apply H to every qubit — all bits are independently random
    qc.h(range(num_bits))
    qc.measure(qr, cr)
    return qc


def generate_random_number(num_bits: int = 8) -> int:
    """
    Generate a single quantum random integer in [0, 2^num_bits - 1].

    Args:
        num_bits: Bit-width of the random number.

    Returns:
        A random integer derived from quantum measurement.
    """
    qc = build_qrng_circuit(num_bits)
    simulator = AerSimulator()
    job = simulator.run(qc, shots=1)
    result = job.result()
    counts = result.get_counts(qc)

    # The single measured bitstring (e.g. '10110011')
    bitstring = list(counts.keys())[0]
    return int(bitstring, 2)


def generate_random_in_range(min_val: int, max_val: int) -> int:
    """
    Generate a quantum random integer in [min_val, max_val].
    Uses rejection sampling to avoid modulo bias.

    Args:
        min_val: Lower bound (inclusive).
        max_val: Upper bound (inclusive).

    Returns:
        Unbiased random integer in [min_val, max_val].
    """
    span = max_val - min_val + 1
    num_bits = math.ceil(math.log2(span)) if span > 1 else 1

    while True:
        raw = generate_random_number(num_bits)
        if raw < span:
            return min_val + raw


def run_uniformity_test(num_bits: int = 4, num_samples: int = 2000) -> None:
    """
    Generate many random numbers and plot their distribution to verify uniformity.

    Args:
        num_bits: Bit width (range: 0 to 2^num_bits - 1).
        num_samples: How many numbers to generate via batch simulation.
    """
    qc = build_qrng_circuit(num_bits)
    simulator = AerSimulator()
    job = simulator.run(qc, shots=num_samples)
    result = job.result()
    counts = result.get_counts(qc)

    # Convert bitstrings to integers
    int_counts = {int(k, 2): v for k, v in counts.items()}
    max_val = 2 ** num_bits - 1

    # Fill missing values with 0
    full_range = {i: int_counts.get(i, 0) for i in range(max_val + 1)}
    expected = num_samples / (max_val + 1)

    print("=" * 50)
    print("  QUANTUM RANDOM NUMBER GENERATOR — UNIFORMITY TEST")
    print("=" * 50)
    print(f"  Bits per sample  : {num_bits}")
    print(f"  Range            : 0 – {max_val}")
    print(f"  Total samples    : {num_samples}")
    print(f"  Expected per bin : {expected:.1f}")
    print()
    print(f"  {'Value':>6}  {'Count':>6}  {'Bar'}")
    print(f"  {'─'*6}  {'─'*6}  {'─'*20}")
    for val, cnt in sorted(full_range.items()):
        bar = "█" * int(cnt / num_samples * 80)
        print(f"  {val:>6}  {cnt:>6}  {bar}")
    print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(full_range.keys(), full_range.values(), color="#4A90D9", edgecolor="#2C5F8A", width=0.7)
    ax.axhline(expected, color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Expected ({expected:.0f})")
    ax.set_title(f"QRNG Uniformity — {num_bits}-bit, {num_samples} samples", fontsize=14)
    ax.set_xlabel("Random Number (integer)")
    ax.set_ylabel("Frequency")
    ax.legend()
    plt.tight_layout()
    plt.savefig("qrng_uniformity.png", dpi=150)
    plt.show()
    print("Plot saved to qrng_uniformity.png")


if __name__ == "__main__":
    # --- Single number generation ---
    print("\nCircuit diagram (8-bit QRNG):")
    qc = build_qrng_circuit(8)
    print(qc.draw(output="text"))
    print()

    single = generate_random_number(num_bits=8)
    print(f"Single 8-bit quantum random number : {single}  (range 0–255)")

    bounded = generate_random_in_range(1, 100)
    print(f"Quantum random number in [1, 100]  : {bounded}")
    print()

    # --- Uniformity test ---
    run_uniformity_test(num_bits=4, num_samples=2000)
