# -*- coding: utf-8 -*-
"""
File: DesignMatrix.py
Purpose: Implement a class to make a design matrix for regression
         Called by class Reg (where inputs are verified)
Author: H. Seltman
Date: Dec. 2017
"""
import numpy as np
import pandas as pd


class DesignMatrix():
    """
    Convert formula and DataFrame to a design matrix
    Input: 'formula' is a str of the form "y~x1+x2"
           'data' is a DataFrame containing all of the variables
             in 'formula'
    Limitations: formula RHS is "+" between numeric or categorical variables
    Implementation details:
        1) int is converted to float
        2) 'reg' is an object with specific attributes; a Reg class object
           will suffice
    Goal: compute DesignMatrix.X, supplemented by DesignMatrix.baseline,
          and DesignMatrix.levels.
    """

    def __init__(self, reg):
        self.reg = reg
        self.baselines = {}

    def __repr__(self):
        return "DesignMatrix({0}, Data: {1} x {2})".format(
            self.reg.formula, self.reg.nrow, self.reg.ncol)

    def recode(self, var):
        """ Recode from Series to numpy array
            float is unchanged
            int is converted to float
            others are treated as factors
        """
        if self.reg.data[var].dtype in ('float32', 'float64'):
            return self.reg.data[var].as_matrix().reshape(self.reg.nrow, 1)
        elif self.reg.data[var].dtype in ('int', 'int64'):
            return self.reg.data[var].astype(float).as_matrix().\
                reshape(self.reg.nrow, 1)
        else:
            x = [str(v) for v in self.reg.data[var]]
            if self.reg.strip:
                x = [v.strip() for v in x]
            if self.reg.toupper:
                x = [v.upper() for v in x]
            if self.reg.tolower:
                x = [v.lower() for v in x]
            if self.baselines.get(var) is None:
                self.baselines[var] = min(x)

            cnts = pd.Series(x).value_counts()
            names = sorted(cnts.index)
            temp = names.copy()
            temp.remove(self.baselines[var])
            self.levels[var] = temp
            X = np.full((self.reg.nrow, len(cnts) - 1), np.nan)
            offset = 0
            for i in range(len(cnts)):
                if names[i] == self.baselines[var]:
                    offset = -1
                    next
                X[:, i + offset] = [int(v == names[i]) for v in x]
            return(X)

    def make_X(self):
        """ Make design matrix (numpy array) X from IVs """
        self.baselines = self.reg.custom_baselines
        self.levels = {}
        self.X = np.ones((self.reg.nrow, 1))
        for iv in self.reg.IVs:
            self.X = np.concatenate((self.X, self.recode(iv)), 1)


if __name__ == "__main__":
    dat = pd.DataFrame({'age': (33, 44, 55, 44, 33, 22, 33),
                        'gender': ('M', 'm', 'f', 'f', 'F', 'M', 'F'),
                        'y': (12.5, 6.9, 15.2, 13, 15, 17.7, 21.2),
                        'tx': ('p', 'p', 'a', 'a', 'b', 'b', 'b')})

    class Fake():
        def __init__(self, formula, data):
            self.formula = formula
            self.data = data
            self.nrow = len(data)
            self.ncol = data.shape[1]
            self.DV = 'y'
            self.IVs = ['age', 'gender', 'tx']
            self.strip = True
            self.toupper = True
            self.tolower = False
            self.custom_baselines = {'tx': 'P'}

    f = Fake("y~age+gender+tx", dat)
    d = DesignMatrix(f)
    d.make_X()
    print(d.X)
    print(d.baselines)
    print(d.levels)

    f.custom_baselines = {'gender': 'M'}
    f.toupper = False
    d.make_X()
    print(d.X)
    print(d.baselines)
    print(d.levels)

    f.custom_baselines = {'gender': 'm'}
    f.tolower = True
    d.make_X()
    print(d.X)
    print(d.baselines)
    print(d.levels)
