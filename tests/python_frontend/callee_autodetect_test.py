# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests automatic detection and parsing of nested functions and methods that are
not annotated with @dace decorators.
"""
import dace
from dace.frontend.python.common import DaceSyntaxError
from dataclasses import dataclass
import numpy as np
import pytest

@dataclass
class SomeClass:
    q: float

    def method(self, a):
        return a * self.q

    def __call__(self, a):
        return self.method(a)

def freefunction(A):
    return A + 1


def test_autodetect_function():
    """ 
    Tests auto-detection of parsable free functions in the Python frontend.
    """
    @dace
    def adf(A):
        return freefunction(A)

    A = np.random.rand(20)
    B = adf(A)
    assert np.allclose(B, A + 1)


def test_autodetect_method():
    obj = SomeClass(0.5)

    @dace
    def adm(A):
        return obj.method(A)

    A = np.random.rand(20)
    B = adm(A)
    assert np.allclose(B, A / 2)


def test_autodetect_callable_object():
    obj = SomeClass(0.5)

    @dace
    def adco(A):
        return obj(A)

    A = np.random.rand(20)
    B = adco(A)
    assert np.allclose(B, A / 2)


def test_nested_function_method():
    @dataclass
    class TestClass:
        some_field: int

        def some_method(self, q):
            return q * self.some_field

    obj = TestClass(5)

    def nested(a):
        return a + 1 + obj.some_method(a)

    @dace
    def nfm(a: dace.float64[20]):
        return nested(a)

    A = np.random.rand(20)
    ref = nfm.f(A)
    daceres = nfm(A)
    assert np.allclose(ref, daceres)


def test_function_that_needs_replacement():
    @dace
    def notworking(a: dace.float64[20]):
        return np.allclose(a, a)

    A = np.random.rand(20)
    with pytest.raises(DaceSyntaxError):
        notworking(A)


if __name__ == '__main__':
    test_autodetect_function()
    test_autodetect_method()
    test_autodetect_callable_object()
    test_nested_function_method()
    test_function_that_needs_replacement()
