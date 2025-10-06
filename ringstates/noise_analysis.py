import itertools as iter
import matplotlib.pyplot as plt
import numpy as np
import time

from photonic_circuit_solver import Stabilizer, qiskit_circuit_solver
from qiskit.quantum_info import Statevector, state_fidelity
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


def noise_analysis(num_photons: int, ordering: list[list[int]], prob_cnot: float, 
                   T_ratio: float, method: str = "density_matrix", 
                   shots: int = 1024, timer: bool = False) -> tuple[any, float]:
    """
    Simple error analysis for a given ordering's qiskit circuit.

    Parameters
    ----------
    num_photons : int
        Number of photons used in graph state
    ordering : list[list[int]]
        Graph state edgelist, index starts at 0
    prob_cnot : float
        Probability for an error to occur after a CNOT
    T_ratio : float
        Ratio of single qubit gate time to CNOT gate time
    method : str
        Method for simulating the circuit, matrix_product_state is faster but noisy, 
        density_matrix (default) is slower but accurate
    shots : int
        Number of times to run the circuit for matrix_product_state method, default is
        1024 shots
    timer : bool
        Times each function, used for optimizing, defaults to false.

    Returns
    -------
    result : dict
        Dictionary with relevant data on simulation.
    fidelity : float
        Fidelity of the noisy density matrix and blank state.
    qc : 
        Qiskit circuit
    """

    if prob_cnot > 0.5:
        raise ValueError("Input lower probability value (p <= 0.5)")

    time_start = time.perf_counter()
        
    qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
    photons = range(num_photons)
    emitters = range(num_photons, qc.num_qubits)

    #inverse of generation circuit
    for edge in ordering:
        qc.cz(edge[0], edge[1])
    for photon in photons:
        qc.h(photon)
    if method == "density_matrix":
        qc.save_state()
    else:
        for photon in photons:
            qc.measure(photon, photon) #measures qubit n to ancilla n

    noise_model = NoiseModel()

    #categorizes probability by gate time
    prob_single = 0.5 * (1 - (1 - 2 * prob_cnot) ** T_ratio) #probability for single qubit gates & emissions

    hadamard_error = pauli_error([("X", prob_single), ("I", 1 - prob_single)]) #z propagated past hadamard
    phase_error = pauli_error([("Z", prob_single), ("I", 1 - prob_single)]) #z propagated past phase
    emission_cx_error = pauli_error([("IZ", prob_single), ("II", 1 - prob_single)]) #z originating on control
    emitter_cx_error = pauli_error([("IZ", prob_cnot), ("II", 1 - prob_cnot)]) #z originating on control

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

        blank_state = [0] * 2**qc.num_qubits
        blank_state[0] = 1
        blank_state = Statevector(blank_state)

        fidelity = state_fidelity(blank_state, noisy_density_matrix)
    
    else:
        counts = result.get_counts(qc)
        try:
            blank_counts = counts['0'*qc.num_qubits]
        except:
            blank_counts = 0

        fidelity = blank_counts / sum(list(counts.values()))
    
    time_end = time.perf_counter()

    print(f"Time taken (qiskit): {round((time_end-time_start) * 1000, 3)} ms") if timer else None

    return result, fidelity


def contour_plot(num_photons: int, ordering: list[list[int]], 
                 t_max: float, t_int: float, p_max: float, p_int: float) -> None:
    """
    Generates a contour plot of the gate timing ratio, probability for CNOT,
    and fidelity,

    Parameters
    ----------
    num_photons : int
        Number of photons used in graph state
    ordering : list[list[int]]
        Graph state edgelist, index starts at 0
    t_max : float
        Maximum gate timing ratio
    t_max : float
        Interval of gate timing ratio
    p_max : float
        Maximum probability for CNOT
    t_max : float
        Interval of probability for CNOT
    """

    ratios = np.round(np.arange(0, t_max+t_int, t_int), 6)
    probs = np.round(np.arange(0, p_max+p_int, p_int), 6)

    data = []
    for t in ratios:
        fidelities = []
        for p in probs:
            result, fidelity = noise_analysis(num_photons, ordering, p, t)
            fidelities.append(round(fidelity, 3))
        data.append(fidelities)

    plt.contourf(probs, ratios, data, levels=list(np.arange(0, 1.1, 0.1)))
    plt.xlabel("Probability of CNOT Error")
    plt.ylabel("T[Single] / T[CNOT]")
    plt.title(f"Fidelities of a {num_photons} Ring State")
    plt.colorbar()
    plt.show()