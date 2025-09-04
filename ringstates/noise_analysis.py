import itertools as iter

import time
import qiskit

from photonic_circuit_solver import *
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

def noise_analysis(num_photons:int, ordering:list, prob_cnot:float, method:str = "density_matrix", shots:int=1024, timer:bool = False):
    """
    Simple error analysis for a given ordering's qiskit circuit.

    Parameters:
    - num_photons : Number of photons used in graph state
    - ordering : Graph state edgelist (Photon index must start at 0)
    - probability : Probability for a cnot error

    Returns:
    - result : Dictionary with relevant data on simulation.
    - fidelity : Fidelity of the noisy density matrix and blank state.
    - qc : Qiskit circuit
    """

    time_start = time.time()

    if prob_cnot > 0.5:
        print("Input lower probability value")
        return (None, None)
        
    qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
    photons = range(num_photons)
    emitters = range(num_photons, qc.num_qubits)

    #sets ancilla 0 to computational 0 state
    for emitter in emitters:
        qc.measure(emitter, 0)
    #inverse of generation circuit
    for edge in ordering:
        qc.cz(edge[0], edge[1])
    for photon in photons:
        qc.h(photon)
    if method == "density_matrix":
        qc.save_state()
    else:
        for photon in photons:
            qc.measure(photon, photon+1)

    noise_model = NoiseModel()

    #categorizes probability by gate time
    T_ratio = 0.1
    prob_single = 0.5 * (1 - (1 - 2 * prob_cnot) ** T_ratio) #probability for single qubit gates & emissions

    hadamard_error =    pauli_error([("X", prob_single), ("I", 1 - prob_single)]) #z propagated past hadamard
    phase_error =       pauli_error([("Y", prob_single), ("I", 1 - prob_single)]) #z propagated past phase
    emission_cx_error = pauli_error([("IZ", prob_single), ("II", 1 - prob_single)]) #z originating on control
    emitter_cx_error =  pauli_error([("IZ", prob_cnot), ("II", 1 - prob_cnot)]) #z originating on control

    for emitter in emitters:
        noise_model.add_quantum_error(hadamard_error, "h", [emitter])
        noise_model.add_quantum_error(phase_error, "s", [emitter])
        for photon in photons:
            noise_model.add_quantum_error(emission_cx_error, "cx", [emitter, photon])

    if len(emitters) >= 2:
        emitter_perms = iter.permutations(emitters, 2)
        for perm in emitter_perms:
            noise_model.add_quantum_error(emitter_cx_error, "cx", perm)

    simulator = AerSimulator(method=method, noise_model=noise_model)
    simulator.set_option("shots", 1 if method == "density_matrix" else shots)
    
    result = simulator.run(qc).result()

    if method == "density_matrix":
        noisy_density_matrix = result.data(0)["density_matrix"]

        blank_state = [0] * 2 ** qc.num_qubits
        blank_state[0] = 1
        blank_state = qiskit.quantum_info.Statevector(blank_state)

        fidelity = qiskit.quantum_info.state_fidelity(blank_state, noisy_density_matrix)
    
    else:
        counts = result.get_counts(qc)
        try:
            blank_counts = counts['0'*qc.num_qubits]
        except:
            blank_counts = 0

        fidelity = blank_counts / sum(list(counts.values()))

    print(f"Time taken (qiskit): {round((time.time()-time_start) * 1000, 3)} ms") if timer else None

    return result, fidelity, qc

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    None