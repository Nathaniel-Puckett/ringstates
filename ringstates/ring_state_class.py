import itertools as iter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import time

from collections import Counter
from copy import deepcopy
from math import factorial
from photonic_circuit_solver import Stabilizer, qiskit_circuit_solver
from random import shuffle


class RingState:
    """
    Class for working with an n ring state in the context of photonic quantum computing.
    """

    def __init__(self, nodes: int, timer: bool = False):
        """
        Parameters
        ----------
        nodes : int
            Number of nodes the graph state contains.
        timer : bool
            Times each method, used for optimizing, defaults to false.
        """

        self.nodes = nodes
        self.orderings = list()
        self.data = list()
        self.timer = timer

    def __getitem__(self, index: int) -> list[list[int]]:
        """
        Returns ordering at index.

        Parameters
        ----------
        index : int
            Index of desired ordering from list of unique orderings.

        Returns
        -------
        self.orderings[index] : list[list[int]]
            Ordering at index, returns none if an ordering does not exist at the index.
        """

        return self.orderings[index]

    def get_all_orderings(self) -> None:
        """
        Generates all possible orderings for the ring state.
        """

        time_start = time.perf_counter()

        self.orderings = list()

        permutations = [list(perm) for perm in iter.permutations(range(self.nodes)[1:], self.nodes-1)]
        for permutation in permutations:
            if permutation[0] == self.nodes-1: #reflective symmetry check 1
                break
            elif permutation[0] > permutation[-1]: #reflective symmetry check 2
                pass
            else:
                permutation.insert(0, 0)
                ordering = [[permutation[i], permutation[i+1]] for i in range(self.nodes-1)]
                ordering.append([permutation[-1], permutation[0]])
                self.orderings.append(ordering)
        
        time_end = time.perf_counter()
        print(f"Time taken (orderings): {round((time_end-time_start) * 1000, 3)} ms") if self.timer else None

    def get_random_orderings(self, num_orderings: int) -> None:
        """
        Generates a number of possible orderings for the ring state.
        
        Parameters
        ----------
        num_orderings : int
            The number of random orderings to generate.
        """
        
        time_start = time.perf_counter()

        self.orderings = list()

        if num_orderings > factorial(self.nodes-1) / 2:
            num_orderings = factorial(self.nodes-1) / 2

        while len(self.orderings) < num_orderings:
            permutation = list(range(self.nodes)[1:])
            shuffle(permutation)
            if permutation[0]>permutation[-1] or permutation[0]==self.nodes-1: #reflective symmetry check
                pass
            else:
                permutation.insert(0, 0)
                ordering = [[permutation[i], permutation[i+1]] for i in range(self.nodes-1)]
                ordering.append([permutation[-1], permutation[0]])
                self.orderings.append(ordering)

        time_end = time.perf_counter()
        print(f"Time taken (orderings): {round((time_end-time_start) * 1000, 3)} ms") if self.timer else None

    def add_ordering(self, ordering: list[list[int]]) -> None:
        """
        Adds an ordering to the list of orderings

        Parameters
        ----------
        ordering : list[list[int]]
            Node ordering to use
        """
        
        self.orderings.append(ordering)

    def generate_data(self, ordering: list[list[int]], index: int) -> list[list]:
        """ 
        Generates relevant data for a given ordering.

        Parameters
        ----------
        ordering : list[list[int]]
            Node ordering to use.
        index : int
            Index of ordering.

        Returns
        -------
        ordering_data : list[list]
            Nested list containing relevant data.
        """

        qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
        qcd = dict(qc.count_ops())
        ordering_data = [
            ["Index", index], 
            ["# CNOT", (qcd.get('cx') - self.nodes) if qcd.get('cx') else 0], 
            ["# Hadamard", qcd.get('h') if qcd.get('h') else 0], 
            ["# Phase", qcd.get('s') if qcd.get('s') else 0], 
            ["Depth", qc.depth()], 
            ["# Emitter", qc.num_qubits - self.nodes]
            ]
        
        return ordering_data

    def get_lowest(self, max: int = 0) -> int:
        """ 
        Finds the ordering with the least number of CNOTs and Hadamards and the lowest depth.

        Parameters
        ----------
        max : int
            Maximum index to search up to, default value is 0 (all indices).

        Returns
        -------
        l_index : int
            Index of the ordering containing the least number of CNOTs, Hadamards, and depth.
        """

        time_start = time.perf_counter()

        self.get_all_orderings() if not self.orderings else None
        max = len(self.orderings) if not max else max

        for index, ordering in enumerate(self.orderings[0:max]):
            ord_data = self.generate_data(ordering, index)
            
            if index == 0:
                l_index = index
                l_data = ord_data
            else:
                for i in range(1, 5):
                    if ord_data[i][1] < l_data[i][1]:
                        print(f"Previous lowest index : {l_index} | New lowest index : {index}")
                        l_index = index
                        l_data = ord_data
                        break
                    elif ord_data[i][1] > l_data[i][1]:
                        break
                
            self.data.append(ord_data)
        
        time_end = time.perf_counter()
        print(f"Time taken (lowest): {round((time_end-time_start) * 1000, 3)} ms") if self.timer else None

        return l_index

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

    def circuit(self, index: int):
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

    def plot(self, x_index: int = 1, y_index: int = 2):
        """
        Plots data from two indicies as a pyplot.

        Parameters
        ----------
        x_index : int
            The index of the values to use for the x axis.
        y_index : int
            The index of the values to use for the y axis.
        """

        plot_data = deepcopy(self.data)
        plot_data = np.array(plot_data)

        x_data = [int(i) for i in plot_data[:,x_index][:,1]]
        y_data = [int(i) for i in plot_data[:,y_index][:,1]]

        zip_data = tuple(zip(x_data, y_data))

        size = dict(Counter(zip_data))
        coords = list(size.keys())
        coords = np.array(coords)

        plt.scatter(coords[:,0], coords[:,1], s=list(size.values()))
        plt.xlabel(plot_data[0][x_index][0])
        plt.ylabel(plot_data[0][y_index][0])
        plt.title(f"{self.nodes} Ring State Data")
        plt.show()