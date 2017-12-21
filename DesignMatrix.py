# -*- coding: utf-8 -*-
"""
File: DesignMatrix.py
Purpose: Implement a class to make a design matrix for regression
Author: H. Seltman
Date: Dec. 2017
"""
import numpy as np
import pandas as pd
import scipy.stats as ss

class DesignMatrix():
    """
    Convert formula and DataFrame to a design matrix
    Input: 'formula' is a str of the form "y~x1+x2"
           'data' is a DataFrame containing all of the variables
             in 'formula'
    Limitations: formula RHS is "+" between numeric or categorical variables
    Implementation detail: int is converted to float
    """

    def __init__(self, formula, data, strip=True, toupper=True, tolower=False,
                 baselines={}):
        if not isinstance(formula, str):
            raise(TypeError("'formula' must be a 'str'"))
        self.formula = formula
        if not isinstance(data, pd.core.frame.DataFrame):
            raise(TypeError("'data' must be a pandas 'DataFrame'"))
        self.data = data
        self.nrow = len(data)
        self.ncol = data.shape[1]
        self.extractDV()
        if not isinstance(strip, bool):
            raise(TypeError("'strip' must be boolean"))
        self.strip = strip
        if not isinstance(toupper, bool):
            raise(TypeError("'toupper' must be boolean"))
        self.toupper = toupper
        if not isinstance(tolower, bool):
            raise(TypeError("'tolower' must be boolean"))
        self.tolower = tolower
        if not isinstance(baselines, dict):
            raise(TypeError("'base' must be a dictionary"))
        self.baselines = baselines
        self.extractIVs()
        self.levels = {}
        self.makeX()
        self.fit()

    def __repr__(self):
        return "DesignMatrix({0}, Data: {1} x {2})".format(
            self.formula, self.nrow, self.ncol)

    def recode(self, var):
        """ Recode from Series to numpy array
            float is unchanged
            int is converted to float
            others are treated as factors
        """
        if self.data[var].dtype in ('float32', 'float64'):
            return self.data[var].as_matrix().reshape(self.nrow, 1)
        elif self.data[var].dtype in ('int', 'int64'):
            return self.data[var].astype(float).as_matrix().\
                reshape(self.nrow, 1)
        else:
            x = [str(v) for v in self.data[var]]
            if self.strip:
                x = [v.strip() for v in x]
            if self.toupper:
                x = [v.upper() for v in x]
            if self.tolower:
                x = [v.lower() for v in x]
            if self.baselines.get(var) is None:
                self.baselines[var] = min(x)
            elif self.baselines[var] not in x:
                raise(Exception(self.baselines[var] + " is not in " + var))

            cnts = pd.Series(x).value_counts()
            names = sorted(cnts.index)
            temp = names.copy()
            temp.remove(self.baselines[var])
            self.levels[var] = temp
            X = np.full((self.nrow, len(cnts) - 1), np.nan)
            offset = 0
            for i in range(len(cnts)):
                if names[i] == self.baselines[var]:
                    offset = -1
                    next
                X[:, i + offset] = [int(v == names[i]) for v in x]
            return(X)

    def extractDV(self):
        """ Get DV from 'formula' and put in self.DV """
        tilde = self.formula.find("~")
        if tilde == -1:
            raise(Exception("No tilde in formula"))
        self.DV = self.formula[:tilde]
        if self.DV not in self.data.columns:
            raise(Exception("DV from 'formula' not in 'data'"))

    def extractIVs(self):
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

    def makeX(self):
        """ Make design matrix (numpy array) X from IVs """
        self.X = np.ones((self.nrow, 1))
        for iv in self.IVs:
            self.X = np.concatenate((self.X, self.recode(iv)), 1)
        self.p = self.X.shape[1]

    def fit(self):
        vcov_unadj = np.linalg.inv(self.X.T @ self.X)
        bhat = vcov_unadj @ self.X.T @ self.data[self.DV].values
        bnames = ['Intercept']
        for iv in self.IVs:
            if iv in self.levels:
                levels = self.levels[iv]
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
    dat = pd.DataFrame({'age': (33, 44, 55, 44, 33, 22, 33),
                        'gender': ('M', 'm', 'f', 'f', 'F', 'M', 'F'),
                        'y': (12.5, 6.9, 15.2, 13, 15, 17.7, 21.2),
                        'tx': ('p', 'p', 'a', 'a', 'b', 'b', 'b')})
    d = DesignMatrix("y~age+gender+tx", dat)
    print(d.data)
    print(d.recode('gender'))
    print(d.recode('tx'))
    print(d.baselines)
    print(d.levels)
    print(d.recode('age'))
    print(d.recode('y'))
    print(d.X)
