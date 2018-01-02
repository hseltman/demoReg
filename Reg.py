#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:13:51 2017

@author: hseltman
"""

import pandas as pd
import DesignMatrix
import numpy as np
import scipy.stats as ss
# import logLike


class Reg:
    """
    Perform regression from an R-style formula and a pandas DataFrame.

    Use a DesignMatrix class object to compute the design matrix.

    Currently only "+" is allowed on the RHS of the formula.  Integer columns
    are converted to float, and 'str' columns are interpreted as factors.
    Calls to the methods of the DesignMatrix object allow variations in
    cleanup of the text in the factors ('str' columns).

    Future versions of DesignMatrix may incorporate support for more complex
    formulas.
    """

    def __init__(self, formula, data):
        """ Check and store inputs"""
        if not isinstance(formula, str):
            raise(TypeError("'formula' must be a 'str'"))
        self.formula = formula.replace(" ", "")
        if not isinstance(data, pd.core.frame.DataFrame):
            raise(TypeError("'data' must be a pandas 'DataFrame'"))
        self.data = data
        self.nrow = len(data)
        self.ncol = data.shape[1]
        self.DesignMat = DesignMatrix.DesignMat(self.formula,
                                                self.data)

        # get DV and IVs
        self.DV = self.DesignMat.DV
        self.IVs = self.DesignMat.IVs

        # Initialize options for string to factor handling
        self.strip = True
        self.toupper = True
        self.tolower = True
        self.custom_baselines = {}
        self.X = None

    def __repr__(self):
        """ Formal, unambiguous class represention """
        return("Reg(formula:" + self.formula + ", size: " +
               str(len(self.data)) + " rows and " +
               str(self.data.shape[1]) + " columns)")

    def __str__(self):
        """ Informal "pretty" class represention """
        return("Reg object:\nFormula: " + self.formula + "\nSize: " +
               str(len(self.data)) + " rows and " + str(self.data.shape[1]) +
               " columns)")

    def make_X(self):
        """ Based on current settings of DesignMatrix, compute 'X' """
        self.DesignMat.make_X()
        self.X = self.DesignMat.X
        self.p = self.X.shape[1]

    def fit(self):
        """ Fit the model from 'X' and the DV.  Store results in  'bhat' and
            other variables.
        """
        self.make_X()
        vcov_unadj = np.linalg.inv(self.X.T @ self.X)
        bhat = vcov_unadj @ self.X.T @ self.data[self.DV].values
        bnames = ['Intercept']
        for iv in self.IVs:
            if iv in self.DesignMat.levels:
                levels = self.DesignMat.levels[iv]
                bnames = bnames + [iv + "." + L for L in levels]
            else:
                bnames = bnames + [iv]
        self.bhat = pd.DataFrame({'estimate': bhat}, index=bnames)
        self.fitted = (self.X @ self.bhat).flatten()
        self.residual = self.data[self.DV].values - self.fitted
        self.SSR = sum([r*r for r in self.residual])
        self.df = self.nrow - self.p
        self.se_residual = (self.SSR / self.df)**0.5
        self.vcov = vcov_unadj * self.se_residual * self.se_residual
        self.bhat = pd.concat((self.bhat,
                               pd.Series([s**0.5 for s in np.diag(self.vcov)],
                                         name='se', index=bnames)), axis=1)
        self.bhat = pd.concat((self.bhat,
                               pd.Series([e/s for (e, s) in
                                          zip(self.bhat['estimate'],
                                              self.bhat['se'])],
                                         name='t', index=bnames)), axis=1)
        self.bhat = pd.concat((self.bhat,
                               pd.Series([(1 - ss.t.cdf(abs(t), self.df)) * 2
                                          for t in self.bhat['t']],
                                         name='p_value', index=bnames)),
                              axis=1)


if __name__ == "__main__":
    dat = pd.DataFrame({'age': [25, 30, 35, 40],
                        'male': ['m', 'M', 'f', 'F'],
                        'score': [45, 52, 88, 51]})
    r = Reg("score ~ age + male", dat)
    print(r.formula)
    print(r.data)
    print(r)
    repr(r)
    print("")
    r.fit()
    print(r.bhat)
    print("")
    r.DesignMat.show_factor_info()
