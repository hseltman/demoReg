# -*- coding: utf-8 -*-
"""
Unit testing of class DesignMat
@author: hseltman
"""


import pytest
import pandas as pd
import numpy as np
import io  # for StringIO
import sys  # for stdout
from demoReg.DesignMatrix import DesignMat


@pytest.fixture(scope="module")
def simpleData():
    """ simple data frame fixture """
    dat = pd.DataFrame({'age': [25, 30, 35, 40],
                        'male': ['m', 'M', 'f ', 'F'],
                        'tx': ['p', 'p', 'a', 'a'],
                        'score': [45, 52, 88, 51]})
    return(dat)


def test_DV_IVs(simpleData):
    """Are DV and IV found?"""
    dm = DesignMat("score ~ age + male", simpleData)
    assert dm.DV == "score"
    assert len(dm.IVs) == 2
    assert dm.IVs == ['age', 'male']


def test_bad_DV(simpleData):
    """Does a bad DV raise an exception?"""
    with pytest.raises(Exception, match="DV from 'formula'"):
        DesignMat("scor ~ age + male", simpleData)


def test_bad_IV(simpleData):
    """Does a bad IV raise an exception?"""
    with pytest.raises(Exception, match="'gender' from"):
        DesignMat("score ~ age + gender", simpleData)


def test_X_made_correctly(simpleData):
    """Is X is a correctly sized array?"""
    dm = DesignMat("score ~ age + male", simpleData)
    dm.make_X()
    assert isinstance(dm.X, np.ndarray)
    assert dm.X.shape == (4, 3)


def test_show_output(simpleData):
    """Test printed output of show_factor_info()"""
    capturedOutput = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = capturedOutput
    dm = DesignMat("score ~ age + male", simpleData)
    dm.make_X()
    dm.show_factor_info()
    sys.stdout = old_stdout
    out = capturedOutput.getvalue().split("\n")
    expect = ["Factor details:", "'male': baseline = 'F'",
              "          others = 'M'", ""]
    assert len(out) == len(expect)
    assert out == expect


def test_no_factor_manipulation(simpleData):
    """Are there five levels when factor manipulation is absent?"""
    dm = DesignMat("score ~ age + male", simpleData)
    dm.set_toupper(False)
    dm.make_X()
    assert dm.X.shape[1] == 5


def test_no_strip(simpleData):
    """Are there four levels when stripping is absent?"""
    dm = DesignMat("score ~ age + male", simpleData)
    dm.set_strip(False)
    dm.set_toupper(True)
    dm.make_X()
    assert dm.X.shape[1] == 4


def test_lowercase(simpleData):
    """Is baseline 'm' with lower case and custom baseline?"""
    dm = DesignMat("score ~ age + male", simpleData)
    dm.set_toupper(False)
    dm.set_tolower(True)
    dm.set_one_baseline('male', 'm')
    dm.make_X()
    assert dm.baselines['male'] == 'm'


def test_custom_baselines(simpleData):
    """Is baseline 'm' with lower case and custom baseline?"""
    dm = DesignMat("score ~ age + male + tx", simpleData)
    dm.set_baselines({'male': 'M', 'tx': 'P'})
    dm.make_X()
    assert dm.baselines['tx'] == 'P'


@pytest.mark.skip(reason="not yet implemented")
def test_explicit_substitute(simpleData):
    dm = DesignMat("score ~ age + male", simpleData)
    dm.pre_substitute('tx', {'p': 'q'})
    dm.make_X()
    assert dm.levels == {'tx': ['Q']}
