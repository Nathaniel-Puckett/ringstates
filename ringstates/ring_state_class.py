import itertools as iter
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from math import factorial, floor
from os import cpu_count
from photonic_circuit_solver import Stabilizer, qiskit_circuit_solver, num_emitters
from qiskit import QuantumCircuit
from random import shuffle
from tqdm import tqdm


def best_state_heuristic(n: int) -> tuple[tuple]:
    """ 
    Heuristic for the best possible ring state in the context of the algorithm used.
    Only applicable for n>=8.

    Parameters
    ----------
    n : int
        Number of nodes

    Returns
    -------
    ordering : tuple[tuple]
        Ordering of the best possible ring state
    """
    ordering = list()

    ordering.append(list(range(0, floor(n/2)-1)))
    ordering.append([n-3, n-1, n-2, n-4])
    ordering.append(list(range(n-5, floor(n/2)-2, -1)))
    ordering.append([0])

    ordering = [element for sublist in ordering for element in sublist]
    ordering = tuple((ordering[i], ordering[i+1]) for i in range(n))

    return ordering


def low_ordering_data(nodes: int, index: int, ordering: tuple[tuple[int]]) -> tuple[tuple]:
    """ 
    Generates data only for orderings with 2 emitters, used for get_lowest_multi method.

    Parameters
    ----------
    nodes : int
        Number of nodes
    index : int
        Index of ordering.
    ordering : tuple[tuple[int]]
        Node ordering to use.

    Returns
    -------
    ordering_data : tuple[tuple]
        Nested list containing relevant data.
    """

    emitters = num_emitters(Stabilizer(edgelist=ordering))
    if emitters > 2:
        return None
    
    qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
    qcd = dict(qc.count_ops())
    ordering_data = (
        ("Index", index), 
        ("# CNOT", (qcd.get('cx') - nodes)), 
        ("# Hadamard", qcd.get('h') if qcd.get('h') else 0), 
        ("# Phase", qcd.get('s') if qcd.get('s') else 0), 
        ("Depth", qc.depth()), 
        ("# Emitter", qc.num_qubits - nodes)
    )
    
    return ordering_data


class RingState:
    """
    Class for working with an n ring state in the context of photonic quantum computing.
    """

    def __init__(self, nodes: int) -> None:
        """
        Parameters
        ----------
        nodes : int
            Number of nodes the graph state contains.
        """

        if nodes < 3:
            raise ValueError("Invalid number of nodes, 3 or more nodes are required to be a valid ring state.")

        self.nodes = nodes
        self.orderings = list()
        self.data = list()
        self.lowest = None
        self.maximum = factorial(self.nodes-1) / 2

    def __getitem__(self, index: int) -> tuple[tuple[int]]:
        """
        Returns ordering at index.

        Parameters
        ----------
        index : int
            Index of desired ordering from list of unique orderings.

        Returns
        -------
        self.orderings[index] : tuple[tuple[int]]
            Ordering at index, returns none if an ordering does not exist at the index.
        """

        return self.orderings[index]
    
    def add_ordering(self, ordering: tuple[tuple[int]]) -> None:
        """
        Adds an ordering to the list of orderings

        Parameters
        ----------
        ordering : tuple[tuple[int]]
            Node ordering to use
        """
        
        self.orderings.append(ordering)

    def get_all_orderings(self, max_num: int = 0) -> None:
        """
        Generates all possible orderings for the ring state up to a maximum value.
        """

        self.orderings = list()
        max_num = max_num if max_num else self.maximum

        permutations = iter.permutations(range(self.nodes)[1:], self.nodes-1)
        pbar = tqdm(total=max_num)

        for permutation in permutations:
            if permutation[0] == self.nodes-1 or len(self.orderings) == max_num: #reflective symmetry check 1
                pbar.close()
                break
            if permutation[0] > permutation[-1]: #reflective symmetry check 2
                continue
            permutation = (0,) + permutation + (0,)
            ordering = tuple((permutation[i], permutation[i+1]) for i in range(self.nodes))
            self.orderings.append(ordering)
            pbar.update(1)

    def get_random_orderings(self, num_orderings: int) -> None:
        """
        Generates a number of possible orderings for the ring state.
        
        Parameters
        ----------
        num_orderings : int
            The number of random orderings to generate.
        """

        self.orderings = list()

        if num_orderings > self.maximum:
            self.get_all_orderings()
            return None

        while len(self.orderings) < num_orderings:
            permutation = list(range(self.nodes)[1:])
            shuffle(permutation)
            permutation = (0,) + tuple(permutation) + (0,)
            if permutation[1]>permutation[-2] or permutation[1]==self.nodes-1: #reflective symmetry check
                continue
            ordering = tuple((permutation[i], permutation[i+1]) for i in range(self.nodes))
            self.orderings.append(ordering)

    def generate_data(self, index: int, ordering: tuple[tuple[int]]) -> tuple[tuple]:
        """ 
        Generates relevant data for a given ordering.

        Parameters
        ----------
        index : int
            Index of ordering.
        ordering : tuple[tuple[int]]
            Node ordering to use.

        Returns
        -------
        ordering_data : tuple[tuple]
            Nested list containing relevant data.
        """

        qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
        qcd = dict(qc.count_ops())
        ordering_data = (
            ("Index", index), 
            ("# CNOT", (qcd.get('cx') - self.nodes)), 
            ("# Hadamard", qcd.get('h') if qcd.get('h') else 0), 
            ("# Phase", qcd.get('s') if qcd.get('s') else 0), 
            ("Depth", qc.depth()), 
            ("# Emitter", qc.num_qubits - self.nodes)
        )
        
        return ordering_data
    
    def get_all_data(self) -> None:
        """ 
        Generates data for all orderings
        """

        None if self.orderings else self.get_all_orderings()
        self.data = list()

        for index, ordering in enumerate(tqdm(self.orderings)):
            ord_data = self.generate_data(index, ordering)
            self.data.append(ord_data)

    def get_lowest(self) -> tuple[tuple]:
        """ 
        Finds the ordering with the least number of CNOTs and Hadamards and the lowest depth.

        Returns
        -------
        lowest_data : tuple[tuple]
            Data of the ordering containing the least number of CNOTs, Hadamards, and depth.
        """

        None if self.orderings else self.get_all_orderings()
        lowest_data = None

        if self.nodes > 6 and len(self.orderings) == self.maximum:
            u_bound = int((2/(self.nodes - 2)) * self.maximum)
        else:
            u_bound = len(self.orderings)

        for index, ordering in enumerate(tqdm(self.orderings[0:u_bound])):
            ordering_data = low_ordering_data(self.nodes, index, ordering)
            if not ordering_data:
                continue
            if not lowest_data:
                lowest_data = ordering_data
            for i in range(1, 5):
                if ordering_data[i][1] < lowest_data[i][1]:
                    lowest_data = ordering_data
                    break
                elif ordering_data[i][1] > lowest_data[i][1]:
                    break

        return lowest_data

    def get_lowest_multi(self, workers: int = None, chunk_size: int = None) -> tuple[tuple]:
        """ 
        Finds the ordering with the least number of CNOTs and Hadamards and the lowest depth.
        Multiprocessing version of get_lowest.

        Parameters
        ----------
        workers : int
            Number of workers to use during multiprocessing, max amount is tied to CPU cores.
            Defaults to the total number of CPU cores.
        chunk_size : int
            Number of orderings to pass to each individual worker at a given time.
            Defaults to u_bound//(cpu_count()*8)

        Returns
        -------
        lowest_data : tuple[tuple]
            Data of the ordering containing the least number of CNOTs, Hadamards, and depth.
        """

        if self.nodes < 9:
            raise ValueError("Value too low (<9) to benefit from multiprocessing, use get_lowest instead")

        None if self.orderings else self.get_all_orderings()
        lowest_data = None

        if self.nodes > 6 and len(self.orderings) == self.maximum:
            u_bound = int((2/(self.nodes - 2)) * self.maximum)
        else:
            u_bound = len(self.orderings)

        workers = workers if workers else cpu_count()
        chunk_size = chunk_size if chunk_size else u_bound//(cpu_count()*8)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            all_ordering_data = list(tqdm(
                                    executor.map(low_ordering_data, [self.nodes]*u_bound, range(0, u_bound), self.orderings, 
                                    chunksize=chunk_size), total = u_bound
                                ))

        for ordering_data in all_ordering_data:
            if not ordering_data:
                continue
            if not lowest_data:
                lowest_data = ordering_data
            for i in range(1, 5):
                if ordering_data[i][1] < lowest_data[i][1]:
                    lowest_data = ordering_data
                    break
                elif ordering_data[i][1] > lowest_data[i][1]:
                    break

        return lowest_data

    def graph(self, index: int) -> nx.Graph:
        """
        Returns the networkx graph for the ordering at index.

        Parameters
        ----------
        index : int
            The index of the ordering to use.

        Returns
        -------
        G : nx.Graph
            Networkx graph of ring state at index.
        """

        G = nx.Graph()
        G.add_nodes_from(range(self.nodes))
        G.add_edges_from(self.orderings[index])
        
        return G

    def circuit(self, index: int) -> QuantumCircuit:
        """
        Returns the circuit for the ordering at index.

        Parameters
        ----------
        index : int
            The index of the ordering to use.

        Returns
        -------
        qc : qiskit circuit
            Circuit representation of ring state at index.
        """

        qc = qiskit_circuit_solver(Stabilizer(edgelist=self.orderings[index]))

        return qc

    def scatterplot(self, x_index: int, y_index: int) -> None:
        """
        Plots data from two indicies as a pyplot.

        Parameters
        ----------
        x_index : int
            The index of the values to use for the x axis.
        y_index : int
            The index of the values to use for the y axis.
        """

        if not self.data:
            raise ValueError('No data to plot, try generating data using get_all_data or get_lowest.')

        plot_data = np.array(self.data)

        x_data = tuple(int(val) for i, val in enumerate(plot_data[:,x_index][:,1]) if plot_data[i][4][1] != '0')
        y_data = tuple(int(val) for i, val in enumerate(plot_data[:,y_index][:,1]) if plot_data[i][4][1] != '0')
        zip_data = tuple(zip(x_data, y_data))

        size = dict(Counter(zip_data))
        coords = list(size.keys())
        coords = np.array(coords)

        plt.scatter(coords[:,0], coords[:,1], s=list(size.values()))
        plt.xlabel(plot_data[0][x_index][0])
        plt.ylabel(plot_data[0][y_index][0])
        plt.title(f"{self.nodes} Ring State Data")
        plt.show()