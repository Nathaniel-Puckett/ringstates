"""
Unit and regression test for the ringstates package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest
import ringstates


def test_ringstates_imported():

    assert "ringstates" in sys.modules

def test_ringstate_class():

    rs = ringstates.RingState(6)
    rs.get_random_orderings(10)

    assert len(rs.orderings) == 10

    rs.add_ordering([[0, 1]])

    assert len(rs.orderings) == 11

    rs.get_all_orderings()

    assert len(rs.orderings) == 60
    assert rs[0] == [[0, 1], [0, 5], [1, 2], [2, 3], [3, 4], [4, 5]]

    lowest = rs.get_lowest()
    
    assert lowest == 5

    rs.nx_plot(5)
    rs.plot_data(1, 2)

    result, fidelity = rs.qiskit_noise_analysis(5, 0.1)

    assert round(fidelity, 3) == 0.59



    
