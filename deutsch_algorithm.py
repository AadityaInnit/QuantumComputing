from dataclasses import dataclass
from typing import Dict, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ─────────────────────────────────────────────
# Oracle metadata
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class OracleSpec:
    key: str
    label: str
    formula: str
    oracle_type: str   # CONSTANT or BALANCED


ORACLES: Dict[str, OracleSpec] = {
    "constant_0": OracleSpec("constant_0", "Constant-0", "f(x)=0", "CONSTANT"),
    "constant_1": OracleSpec("constant_1", "Constant-1", "f(x)=1", "CONSTANT"),
    "identity":   OracleSpec("identity",   "Identity",   "f(x)=x", "BALANCED"),
    "not":        OracleSpec("not",        "NOT",        "f(x)=1-x", "BALANCED"),
}


# ─────────────────────────────────────────────
# Oracle construction
# ─────────────────────────────────────────────

def make_oracle(oracle: str) -> QuantumCircuit:
    """
    Construct the 2-qubit Deutsch oracle U_f acting as:
        U_f |x,y> = |x, y ⊕ f(x)>
    """
    if oracle not in ORACLES:
        raise ValueError(f"Unknown oracle '{oracle}'. Choose from {list(ORACLES)}")

    uf = QuantumCircuit(2, name=f"U_f[{oracle}]")

    if oracle == "constant_0":
        pass

    elif oracle == "constant_1":
        uf.x(1)

    elif oracle == "identity":
        uf.cx(0, 1)

    elif oracle == "not":
        # Flip ancilla iff x = 0
        uf.x(0)
        uf.cx(0, 1)
        uf.x(0)

    return uf


# ─────────────────────────────────────────────
# Deutsch circuit
# ─────────────────────────────────────────────

def build_deutsch_circuit(
    oracle: str,
    measure: bool = True,
    barriers: bool = True
) -> QuantumCircuit:
    """
    Build the full Deutsch algorithm circuit for a chosen oracle.

    Qubit convention:
      q0 = query qubit
      q1 = ancilla qubit
    """
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(1, "c") if measure else None
    qc = QuantumCircuit(qr, cr) if measure else QuantumCircuit(qr)

    # Prepare |0>|1>
    qc.x(qr[1])

    # Prepare |+>|->
    qc.h(qr[0])
    qc.h(qr[1])

    if barriers:
        qc.barrier()

    qc.append(make_oracle(oracle).to_gate(), [qr[0], qr[1]])

    if barriers:
        qc.barrier()

    # Interference step
    qc.h(qr[0])

    if measure:
        qc.measure(qr[0], cr[0])

    return qc


# ─────────────────────────────────────────────
# Exact analysis via statevector
# ─────────────────────────────────────────────

def exact_query_distribution(oracle: str) -> Tuple[Dict[str, float], Statevector]:
    """
    Return exact probabilities for measuring q0 as 0 or 1,
    computed from the final state before measurement.
    """
    qc = build_deutsch_circuit(oracle, measure=False, barriers=False)
    sv = Statevector.from_instruction(qc)

    probs_q0 = sv.probabilities_dict(qargs=[0])

    # Ensure explicit keys
    dist = {
        "0": float(probs_q0.get("0", 0.0)),
        "1": float(probs_q0.get("1", 0.0)),
    }
    return dist, sv


def classify_from_distribution(dist: Dict[str, float], tol: float = 1e-9) -> str:
    if abs(dist["0"] - 1.0) < tol:
        return "CONSTANT"
    if abs(dist["1"] - 1.0) < tol:
        return "BALANCED"
    return "INDETERMINATE"


# ─────────────────────────────────────────────
# Shot-based simulation
# ─────────────────────────────────────────────

def run_shot_simulation(oracle: str, shots: int = 1024) -> Dict[str, int]:
    qc = build_deutsch_circuit(oracle, measure=True, barriers=True)
    sim = AerSimulator()
    tqc = transpile(qc, sim)
    result = sim.run(tqc, shots=shots).result()
    counts = result.get_counts(tqc)
    return dict(counts)


def classify_from_counts(counts: Dict[str, int]) -> str:
    dominant = max(counts, key=counts.get)
    return "CONSTANT" if dominant == "0" else "BALANCED"


# ─────────────────────────────────────────────
# Stepwise statevector walkthrough
# ─────────────────────────────────────────────

def get_statevectors_by_stage(oracle: str) -> Dict[str, Statevector]:
    """
    Produce exact statevectors at key points of the algorithm.
    """
    states = {}

    qc = QuantumCircuit(2)
    qc.x(1)
    states["After init |01>"] = Statevector.from_instruction(qc)

    qc.h(0)
    qc.h(1)
    states["After H⊗H"] = Statevector.from_instruction(qc)

    qc.append(make_oracle(oracle).to_gate(), [0, 1])
    states["After oracle"] = Statevector.from_instruction(qc)

    qc.h(0)
    states["After final H(q0)"] = Statevector.from_instruction(qc)

    return states


def fmt_complex(z: complex, tol: float = 1e-10) -> str:
    r = 0.0 if abs(z.real) < tol else z.real
    i = 0.0 if abs(z.imag) < tol else z.imag

    if i == 0.0:
        return f"{r:+.3f}"
    if r == 0.0:
        return f"{i:+.3f}j"
    return f"{r:+.3f}{i:+.3f}j"


def print_statevector_walkthrough(oracle: str) -> None:
    spec = ORACLES[oracle]
    states = get_statevectors_by_stage(oracle)

    print(f"\nStatevector walkthrough — {spec.label} [{spec.formula}]")
    print(f"{'Step':<22} {'|00>':>12} {'|01>':>12} {'|10>':>12} {'|11>':>12}")
    print("-" * 74)

    for step, sv in states.items():
        amps = [fmt_complex(a) for a in sv.data]
        print(f"{step:<22} {amps[0]:>12} {amps[1]:>12} {amps[2]:>12} {amps[3]:>12}")


# ─────────────────────────────────────────────
# Unified analysis
# ─────────────────────────────────────────────

def analyze_oracle(oracle: str, shots: int = 1024) -> dict:
    spec = ORACLES[oracle]

    exact_dist, sv = exact_query_distribution(oracle)
    exact_verdict = classify_from_distribution(exact_dist)

    counts = run_shot_simulation(oracle, shots=shots)
    shot_verdict = classify_from_counts(counts)

    return {
        "oracle": oracle,
        "label": spec.label,
        "formula": spec.formula,
        "expected": spec.oracle_type,
        "exact_distribution": exact_dist,
        "exact_verdict": exact_verdict,
        "counts": counts,
        "shot_verdict": shot_verdict,
        "statevector": sv,
    }


# ─────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────

def plot_all_oracles(shots: int = 1024, save_path: str = "deutsch_algorithm_refactored.png") -> None:
    oracle_keys = list(ORACLES.keys())
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    palette = {"CONSTANT": "#4A90D9", "BALANCED": "#E74C3C"}

    for ax, oracle in zip(axes, oracle_keys):
        result = analyze_oracle(oracle, shots=shots)
        counts = result["counts"]
        verdict = result["shot_verdict"]

        bars = ax.bar(
            ["0 (CONST)", "1 (BAL)"],
            [counts.get("0", 0), counts.get("1", 0)],
            color=[palette["CONSTANT"], palette["BALANCED"]],
            edgecolor="white",
            width=0.55
        )

        ax.set_title(f"{result['label']}\n[{result['formula']}]", fontsize=10, fontweight="bold")
        ax.set_ylabel("Counts")
        ax.set_ylim(0, shots * 1.12)

        ax.text(
            0.5, 0.92, f"→ {verdict}",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=11, fontweight="bold",
            color=palette[verdict]
        )

        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + shots*0.015, f"{int(h)}",
                        ha="center", va="bottom", fontsize=9)

    const_patch = mpatches.Patch(color=palette["CONSTANT"], label="Measure 0 → CONSTANT")
    bal_patch = mpatches.Patch(color=palette["BALANCED"], label="Measure 1 → BALANCED")

    fig.legend(handles=[const_patch, bal_patch], loc="lower center",
               ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Deutsch's Algorithm — Exact Logic Confirmed by Shot Simulation",
                 fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ─────────────────────────────────────────────
# Main demo
# ─────────────────────────────────────────────

if __name__ == "__main__":
    SHOTS = 1024

    print("=" * 72)
    print("DEUTSCH'S ALGORITHM")
    print("Single-query distinction between CONSTANT and BALANCED Boolean functions")
    print("=" * 72)

    for oracle in ORACLES:
        result = analyze_oracle(oracle, shots=SHOTS)

        print("\n" + "-" * 72)
        print(f"Oracle              : {result['label']} [{result['formula']}]")
        print(f"Expected type       : {result['expected']}")
        print(f"Exact distribution  : {result['exact_distribution']}")
        print(f"Exact verdict       : {result['exact_verdict']}")
        print(f"Shot counts         : {result['counts']}")
        print(f"Shot verdict        : {result['shot_verdict']}")
        print(f"Matches expectation : {result['shot_verdict'] == result['expected']}")

        qc = build_deutsch_circuit(oracle, measure=True, barriers=True)
        print("\nCircuit:")
        print(qc.draw(output="text"))

        print_statevector_walkthrough(oracle)

    print("\n" + "-" * 72)
    print("Key idea:")
    print("Preparing the ancilla in |-> converts the oracle's bit-flip action")
    print("into a phase factor (-1)^f(x) on the query superposition.")
    print("The final Hadamard on q0 turns that relative phase into a measurable bit.")
    print("-" * 72)

    plot_all_oracles(shots=SHOTS)