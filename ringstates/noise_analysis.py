import itertools as iter
import numpy as np

import time
import qiskit

from photonic_circuit_solver import *
from ringstates import *
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


def noise_analysis(rs:RingState, index:int, P_large:float):
    """
    Simple error analysis for a given ordering's qiskit circuit.

    Parameters:
    - rs : RingState class object
    - index : Index of ordering to analyze.
    - P_large : The probability that an error after an cnot between emitters occurs

    Returns:
    - result : Dictionary with relevant data on simulation.
    - fidelity : Fidelity of the noisy density matrix and blank state.
    """

    time_start = time.time()

    if P_large > 1:
        print("Input lower probability value")
        return (None, None)
        
    qc = qiskit_circuit_solver(Stabilizer(edgelist=rs.orderings[index]))

    #sets ancilla 0 to computational 0 state
    for emitter in range(rs.nodes, qc.num_qubits):
        qc.measure(emitter, 0)
    #inverse of generation circuit
    for edge in rs.orderings[index]:
        qc.cz(edge[0], edge[1])
    for photon in range(len(rs.orderings[index])):
        qc.h(photon)
    qc.save_state()
    for photon in range(len(rs.orderings[index])):
        qc.measure(photon, photon+1)

    noise_model = NoiseModel()

    #categorizes probability by gate time
    T_ratio = 0.1
    P_small = (1 - (1 - P_large) ** T_ratio) #probability for single qubit gates & emissions

    hadamard_error = pauli_error([("X", P_small), ("I", 1 - P_small)]) #z propagated past hadamard
    phase_error = pauli_error([("Y", P_small), ("I", 1 - P_small)]) #z propagated past phase
    emission_cx_error = pauli_error([("IZ", P_small), ("II", 1 - P_small)]) #z originating on control
    emitter_cx_error = pauli_error([("IZ", P_large), ("II", 1 - P_large)]) #z originating on control

    for emitter in range(rs.nodes, qc.num_qubits):
        noise_model.add_quantum_error(hadamard_error, "h", [emitter])
        noise_model.add_quantum_error(phase_error, "s", [emitter])
        for photon in range(rs.nodes):
            noise_model.add_quantum_error(emission_cx_error, "cx", [emitter, photon])

    if qc.num_qubits - rs.nodes >= 2:
        emitter_combos = iter.permutations(range(rs.nodes, qc.num_qubits), 2)
        for combo in emitter_combos:
            noise_model.add_quantum_error(emitter_cx_error, "cx", combo)

    simulator = AerSimulator(method='density_matrix', noise_model=noise_model)
    #qc = qiskit.transpile(qc, simulator)
    
    result = simulator.run(qc).result()

    noisy_density_matrix = result.data(0)["density_matrix"]

    blank_state = [0] * 2 ** qc.num_qubits
    blank_state[0] = 1
    blank_state = qiskit.quantum_info.Statevector(blank_state)

    fidelity = qiskit.quantum_info.state_fidelity(blank_state, noisy_density_matrix)

    print(f"Time taken (qiskit): {round((time.time()-time_start) * 1000, 3)} ms") if rs.timer else None

    return result, fidelity

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    None