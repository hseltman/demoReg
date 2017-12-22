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

#import logLike


class Reg:
    def __init__(self, formula, data):
        # Check and store inputs
        if not isinstance(formula, str):
            raise(TypeError("'formula' must be a 'str'"))
        self.formula = formula.replace(" ", "")
        if not isinstance(data, pd.core.frame.DataFrame):
            raise(TypeError("'data' must be a pandas 'DataFrame'"))
        self.data = data
        self.nrow = len(data)
        self.ncol = data.shape[1]
        self.DesignMatrix = DesignMatrix.DesignMatrix(self)

        # Initialize options for string to factor handling
        self.strip = True
        self.toupper = True
        self.tolower = True
        self.custom_baselines = {}

        # get DV and IVs
        self.extract_DV()
        self.extract_IVs()

    # Formal, unambiguous class represention:
    def __repr__(self):
        return("Reg(formula:" + self.formula + ", size: " +
               str(len(self.data)) + " rows and " +
               str(self.data.shape[1]) + " columns)")

    # Informal "pretty" class represention:
    def __str__(self):
        return("Reg object:\nFormula: " + self.formula + "\nSize: " +
               str(len(self.data)) + " rows and " + str(self.data.shape[1]) +
               " columns)")

    def extract_DV(self):
        """ Get DV from 'formula' and put in self.DV """
        tilde = self.formula.find("~")
        if tilde == -1:
            raise(Exception("No tilde in formula"))
        self.DV = self.formula[:tilde]
        if self.DV not in self.data.columns:
            raise(Exception("DV from 'formula' not in 'data'"))

    def extract_IVs(self):
        """ Get IVs from 'formula' and put in self.IVs """
        tilde = self.formula.find("~")
        if tilde == -1:
            raise(Exception("No tilde in formula"))
        RHS = self.formula[tilde + 1:]
        IVs = [x.strip() for x in RHS.split('+')]
        for iv in IVs:
            if iv not in self.data.columns:
                raise(Exception(iv + " from 'formula' not in 'data'"))
        self.IVs = IVs

    def set_baselines(self, replacementBaselineDictionary):
        """
        Explictly set the dictionary of baselines (other than the
        alphabetically first) for categorical variables
        """
        for key in replacementBaselineDictionary.keys():
            if not isinstance(key, str):
                raise(Exception("keys must be 'str' objects"))
            if key not in self.IVs:
                raise(Exception(key + " is not one of the IVs"))
            value = replacementBaselineDictionary[key]
            if not isinstance(value, str):
                raise(Exception("values must be 'str' objects"))
            if value not in self.data[key]:
                raise(Exception(value + " is not in '" + key + "'"))
        self.custom_baselines = replacementBaselineDictionary

    def set_one_baseline(self, var, value):
        """ set or replace a single baseline for a factor """
        if not isinstance(var, str):
            raise(Exception("'var' must be a 'str' object"))
        if var not in self.IVs:
            raise(Exception(var + " is not one of the IVs"))
        if not isinstance(value, str):
            raise(Exception("'value' must be a 'str' object"))
        if value not in self.data[var]:
            raise(Exception(value + " is not in '" + var + "'"))
        self.custom_baseline[var] = value

    def set_strip(self, value):
        if not isinstance(value, bool):
            raise(TypeError("'value' must be a 'bool' object"))
        self.strip = value

    def set_tolower(self, value):
        if not isinstance(value, bool):
            raise(TypeError("'value' must be a 'bool' object"))
        self.tolower = value

    def set_toupper(self, value):
        if not isinstance(value, bool):
            raise(TypeError("'value' must be a 'bool' object"))
        self.toupper = value

    def make_X(self):
        self.DesignMatrix.make_X()
        self.X = self.DesignMatrix.X
        self.p = self.X.shape[1]

    def fit(self):
        self.make_X()
        vcov_unadj = np.linalg.inv(self.X.T @ self.X)
        bhat = vcov_unadj @ self.X.T @ self.data[self.DV].values
        bnames = ['Intercept']
        for iv in self.IVs:
            if iv in self.DesignMatrix.levels:
                levels = self.DesignMatrix.levels[iv]
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
    r.fit()
    print(r.bhat)
    print(r.DesignMatrix.baselines)
    print(r.DesignMatrix.levels)

