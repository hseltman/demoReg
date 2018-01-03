# -*- coding: utf-8 -*-
"""
Test for Reg (py.test)
@author: hseltman
"""

import pytest
from pytest import approx
import pandas as pd
from Reg import Reg
import numpy as np


@pytest.fixture(scope="module")
def simpleData():
    """ simple data frame fixture """
    dat = pd.DataFrame({'age': [25, 30, 35, 40],
                        'male': ['m', 'M', 'f', 'F'],
                        'score': [45, 52, 88, 51]})
    return dat


def test_X_correct(simpleData):
    """Is X is a correctly sized array?"""
    r = Reg("score ~ age + male", simpleData)
    r.make_X()
    assert isinstance(r.X, np.ndarray)
    assert r.X.shape == (4, 3)


def test_fit_correct(simpleData):
    """Does fit() generate correct output?"""
    r = Reg("score ~ age + male", simpleData)
    r.make_X()
    r.fit()
    assert isinstance(r.bhat, pd.DataFrame)
    assert r.bhat.shape == (3, 4)
    assert all(r.bhat.columns.values == ('estimate', 'se', 't', 'p_value'))
    assert all(round(r.bhat['estimate'], 1) == (182.0, -3.0, -51.0))
    assert all(round(r.bhat['se'], 1) == (165.7, 4.4, 49.2))
    assert all(round(r.bhat['t'], 2) == (1.10, -0.68, -1.04))
    assert all(round(r.bhat['p_value'], 2) == (0.47, 0.62, 0.49))
    assert r.SSR == approx(484.0)
    assert r.df == 1
    assert all(r.residual.round(2) == (-11.0, 11.0, 11.0, -11.0))
    assert r.se_residual == approx(22.0)
