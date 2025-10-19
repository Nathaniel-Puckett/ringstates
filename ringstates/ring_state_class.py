import itertools as iter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from collections import Counter
from copy import deepcopy
from graphstates import GraphGen
from math import factorial
from photonic_circuit_solver import Stabilizer, qiskit_circuit_solver
from qiskit import QuantumCircuit
from random import shuffle
from tqdm import tqdm


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

        self.nodes = nodes
        self.orderings = list()
        self.data = list()
        self.maximum = factorial(self.nodes-1) / 2

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
    
    def add_ordering(self, ordering: list[list[int]]) -> None:
        """
        Adds an ordering to the list of orderings

        Parameters
        ----------
        ordering : list[list[int]]
            Node ordering to use
        """
        
        self.orderings.append(ordering)

    def get_all_orderings(self) -> None:
        """
        Generates all possible orderings for the ring state.
        """

        self.orderings = list()

        permutations = [list(perm) for perm in iter.permutations(range(self.nodes)[1:], self.nodes-1)]
        pbar = tqdm(total=self.maximum)

        for permutation in permutations:
            if permutation[0] == self.nodes-1: #reflective symmetry check 1
                pbar.close()
                break
            if permutation[0] > permutation[-1]: #reflective symmetry check 2
                continue
            permutation.insert(0, 0)
            ordering = [[permutation[i], permutation[i+1]] for i in range(self.nodes-1)]
            ordering.append([permutation[-1], permutation[0]])
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
            num_orderings = self.maximum

        while len(self.orderings) < num_orderings:
            permutation = list(range(self.nodes)[1:])
            shuffle(permutation)
            if permutation[0]>permutation[-1] or permutation[0]==self.nodes-1: #reflective symmetry check
                continue
            permutation.insert(0, 0)
            ordering = [[permutation[i], permutation[i+1]] for i in range(self.nodes-1)]
            ordering.append([permutation[-1], permutation[0]])
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
    
    def get_all_data(self) -> None:
        """ 
        Generates data for all orderings
        """

        None if self.orderings else self.get_all_orderings()

        for index, ordering in enumerate(tqdm(self.orderings)):
            ord_data = self.generate_data(ordering, index)
            self.data.append(ord_data)

    def get_lowest(self) -> int:
        """ 
        Finds the ordering with the least number of CNOTs and Hadamards and the lowest depth.

        Returns
        -------
        l_index : int
            Index of the ordering containing the least number of CNOTs, Hadamards, and depth.
        """

        None if self.orderings else self.get_all_orderings()

        if self.nodes > 6 and len(self.orderings) == self.maximum:
            u_bound = int((2/(self.nodes - 2)) * len(self.orderings))
        else:
            u_bound = len(self.orderings)

        for index, ordering in enumerate(tqdm(self.orderings[0:u_bound])):
            g_state = GraphGen(self.graph(index))
            if g_state.num_emitters > 2:
                #all values are set to 0 in order to keep scatterplot method working
                ord_data = [
                    ["Index", index], 
                    ["# CNOT", 0], 
                    ["# Hadamard", 0], 
                    ["# Phase", 0], 
                    ["Depth", 0], 
                    ["# Emitter", 0]
                    ]
                self.data.append(ord_data)
                continue
            ord_data = self.generate_data(ordering, index)
            if index == 0:
                l_index = index
                l_data = ord_data
            for i in range(1, 5):
                if ord_data[i][1] < l_data[i][1]:
                    l_index = index
                    l_data = ord_data
                    break
                elif ord_data[i][1] > l_data[i][1]:
                    break
            self.data.append(ord_data)

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