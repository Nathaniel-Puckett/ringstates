"""
Unit and regression test for the ringstates package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import ringstates


def test_ringstates_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "ringstates" in sys.modules
