import copy as cp
import itertools as iter
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

import time

from collections import Counter
from math import factorial
from photonic_circuit_solver import *
from random import shuffle

class RingState:
    """
    Class for working with and storing information about an n node ring state.
    """

    def __init__(self, nodes:int, timer:bool = False):
        """
        Initializes class.

        Parameters:
        - nodes : Number of nodes the graph state contains.
        """

        self.nodes = nodes
        self.orderings = list()
        self.data = list()
        self.timer = timer
    
    def __getitem__(self, index:int):
        """
        Returns ordering at index.

        Parameters:
        - index : Index of desired ordering from list of unique orderings.

        Returns:
        - self.orderings[index] : Ordering at index, returns none if an ordering does not exist at the index.
        """

        #checks if there are orderings, if not, generates them
        try:
            return self.orderings[index]
        except:
            print("Index out of range")
            return None
    
    def get_all_orderings(self):
        """
        Generates all possible orderings for the ring state.
        
        Explaination:
            Generates all permutations of 1 through n-1, prepending each permuation with 0.
            If a permutation's first nonzero element is greater than the last element, it has already been 
            processed, and is equivalent to some element in the list of orderings.
            Each ordering is created using the form [0, 1], [1, 2], ... [n-1, 0].
        """

        time_start = time.time()

        self.orderings = list()

        permutations = [list(perm) for perm in iter.permutations(range(self.nodes)[1:], self.nodes - 1)]
        for permutation in permutations:
            if permutation[0] == self.nodes-1: #reflective symmetry check 1
                break
            elif permutation[0] > permutation[-1]: #reflective symmetry check 2
                pass
            else:
                permutation.insert(0, 0)
                ordering = []
                for i in range(self.nodes):
                    edge = [permutation[i], permutation[i+1]] if i != self.nodes-1 else [permutation[i], permutation[0]] 
                    ordering.append(edge)
                self.orderings.append(ordering)
        
        print(f"Time taken (orderings): {round((time.time()-time_start) * 1000, 3)} ms") if self.timer else None

    def get_random_orderings(self, num_orderings:int):
        """
        Generates a number of possible orderings for the ring state.
        
        Parameters:
        - num_orderings : The number of random orderings to generate.
        """
        
        time_start = time.time()

        self.orderings = list()

        if num_orderings > factorial(self.nodes-1) / 2:
            num_orderings = factorial(self.nodes-1) / 2

        while len(self.orderings) < num_orderings:
            rd_permutation = list(range(self.nodes)[1:])
            shuffle(rd_permutation)
            if rd_permutation[0] > rd_permutation[-1] or rd_permutation[0] == self.nodes-1: #reflective symmetry check
                pass
            else:
                rd_permutation.insert(0, 0)
                ordering = []
                for i in range(self.nodes):
                    edge = [rd_permutation[i], rd_permutation[i+1]] if i != self.nodes-1 else [rd_permutation[i], rd_permutation[0]] 
                    ordering.append(edge)
                self.orderings.append(ordering)
        
        print(f"Time taken (orderings): {round((time.time()-time_start) * 1000, 3)} ms") if self.timer else None

    def add_ordering(self, ordering:list):
        """
        Adds an ordering to the list of orderings
        """
        
        self.orderings.append(ordering)

    def generate_data(self, ordering:list, index:int):
        """ 
        Generates relevant data for a given ordering.

        Parameters:
        - ordering : Node ordering to use.
        - index : Index of ordering.

        Returns:
        - ordering_data : Nested list containing relevant data.
        """

        qc = qiskit_circuit_solver(Stabilizer(edgelist=ordering))
        qcd = dict(qc.count_ops())
        num_cnot = qcd.get('cx') - self.nodes #only counts cnots between emitters
        num_hadamard = qcd.get('h')
        num_phase = qcd.get('s') if qcd.get('s') else 0
        depth = qc.depth()
        emitters = qc.num_qubits - self.nodes
        ordering_data = [
            ["Index", index], 
            ["# CNOT", num_cnot], 
            ["# Hadamard", num_hadamard], 
            ["# Phase", num_phase], 
            ["# Emitter", emitters], 
            ["Depth", depth]
            ]
        
        return ordering_data

    def get_lowest(self, max=None):
        """ 
        Finds the ordering with the least number of CNOTs and Hadamards and the lowest depth.
        
        Explaination:
            Does a simple search across all orderings, creating a qiskit circuit for each and counting
            the number of CNOTs and Hadamards it consists of. All circuits contain a number of cnots
            equal to the number of nodes, so they are not accounted for.

        Parameters:
        - max : Maximum index to search up to.

        Returns:
        - l_index : Index of the ordering containing the least number of CNOTs, Hadamards, and depth.
        """

        time_start = time.time()

        #checks if there are orderings, if not, generates them
        self.get_all_orderings() if not self.orderings else None

        #checks if a maximum value is specified to search up to, default is all orderings
        max = len(self.orderings) if not max else max

        index = 0

        for ordering in self.orderings[0:max]:
            ord_data = self.generate_data(ordering, index)
            
            if index == 0:
                l_index = index
            else:
                #need to refine checks
                cnot_check = ord_data[1][1] < self.data[l_index][1][1]
                hadamard_check = ord_data[1][1] == self.data[l_index][1][1] and ord_data[2][1] < self.data[l_index][2][1]
                phase_check = ord_data[1][1] == self.data[l_index][1][1] and ord_data[2][1] == self.data[l_index][2][1] and ord_data[3][1] < self.data[l_index][3][1]
                depth_check = ord_data[1][1] == self.data[l_index][1][1] and ord_data[2][1] == self.data[l_index][2][1] and ord_data[3][1] == self.data[l_index][3][1] and ord_data[5][1] < self.data[l_index][5][1]
                
                if cnot_check or hadamard_check or phase_check or depth_check:
                    print(f"Previous lowest index : {l_index} | New lowest index : {index}")
                    l_index = index
                
            self.data.append(ord_data)
            index += 1

        print(f"Time taken (lowest): {round((time.time()-time_start) * 1000, 3)} ms") if self.timer else None

        return l_index
    
    def nx_plot(self, index:int):
        """
        Plots the ordering at index as a networkx graph.

        Parameters:
        - index : The index of the ordering to use.

        Returns:
        - G : Networkx graph of ring state at index.
        """

        G = nx.Graph()
        G.add_nodes_from(range(self.nodes))
        G.add_edges_from(self.orderings[index])
        return G

    def plot_data(self, x_index:int=1, y_index:int=2):
        """
        Plots data from two indicies as a pyplot.

        Parameters:
        - x_index : The index of the values to use for the x axis.
        - y_index : The index of the values to use for the y axis.
        """

        plot_data = cp.deepcopy(self.data)
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
        plt.show()

if __name__ == "__main__":
    # Do something if this file is invoked on its own
    None