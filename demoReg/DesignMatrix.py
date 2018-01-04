# -*- coding: utf-8 -*-
"""
File: DesignMatrix.py
Purpose: Implement a class to make a design matrix for regression
         Called by class Reg (where inputs are verified) or similar
Author: H. Seltman
Date: Dec. 2017
"""
import numpy as np
import pandas as pd


class DesignMat():
    """
    Convert formula and DataFrame to a design matrix
    Input: 'formula' is a str of the form "y~x1+x2"
           'data' is a DataFrame containing all of the variables
             in 'formula'
    Limitations: formula RHS is "+" between numeric or categorical variables
    Implementation details:
        1) int is converted to float
        2) non-numeric columns are coded as factors using "treatment"
           contrasts with the baseline as the alphabetically first level or
           a level set with set_custom_baseline() or set_custom_baselines()
    Usage: First set custom baseline(s) and strip, tolower, & toupper, and
           then run make_X() which constructs 'X'.
    Goal: compute DesignMatrix.X, supplemented by DesignMatrix.baseline,
          and DesignMatrix.levels.
    """

    def __init__(self, formula, data):
        self.formula = formula.replace(" ", "")
        self.data = data
        self.nrow, self.ncol = self.data.shape
        self.extract_DV()
        self.extract_IVs()
        self.strip = True
        self.toupper = True
        self.tolower = False
        self.custom_baselines = {}
        self.baselines = {}
        self.levels = None
        self.max_levels_shown = 10
        self.X = None

    def __repr__(self):
        return "DesignMatrix({0}, Data: {1} x {2})".format(
            self.formula, self.nrow, self.ncol)

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
                raise(Exception("'" + iv + "' from 'formula' not in 'data'"))
        self.IVs = IVs

    def reset_baselines(self):
        self.custom_baselines = {}

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
        self.custom_baselines = replacementBaselineDictionary

    def set_one_baseline(self, var, value):
        """ set or replace a single baseline for a factor """
        if not isinstance(var, str):
            raise(Exception("'var' must be a 'str' object"))
        if var not in self.IVs:
            raise(Exception(var + " is not one of the IVs"))
        if not isinstance(value, str):
            raise(Exception("'value' must be a 'str' object"))
        self.custom_baselines[var] = value

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
        # Code factors (categorical IVs)
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
                print("Baseline '", self.baselines[var], "' is not in '",
                      var, "'.", sep="")
                self.baselines[var] = min(x)
                print("Using '", self.baselines[var], "' instead.", sep="")

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

    def make_X(self):
        """ Make design matrix (numpy array) X from IVs """
        self.baselines = self.custom_baselines
        self.levels = {}
        self.X = np.ones((self.nrow, 1))
        for iv in self.IVs:
            self.X = np.concatenate((self.X, self.recode(iv)), 1)

    def show_factor_info(self):
        if self.X is None:
            print(".make_X() whas not yet been run")
            return
        print("Factor details:")
        for var in self.IVs:
            if var in self.baselines.keys():
                val = self.baselines[var]
                print("'", var, "': baseline = '", val, "'", sep="")
                val = self.levels[var]
                valstr = "'" + "', '".join(val[:self.max_levels_shown]) + "'"
                print(" "*(6 + len(var)) + "others = " + valstr, end="")
                if (len(val) > self.max_levels_shown):
                    print(", ...")
                else:
                    print("")


if __name__ == "__main__":
    dat = pd.DataFrame({'age': (33, 44, 55, 44, 33, 22, 33),
                        'gender': ('M', 'm', 'f ', 'f', 'F', 'M', 'F'),
                        'y': (12.5, 6.9, 15.2, 13, 15, 17.7, 21.2),
                        'tx': ('p', 'p', 'a', 'a', 'b', 'b', 'b')})

    d = DesignMat("y ~ age + gender + tx", dat)
    d.make_X()
    print(d.X)
    d.show_factor_info()

    d.set_toupper(False)
    d.set_strip(False)
    d.set_baselines({'gender': 'M'})
    d.make_X()
    d.show_factor_info()
    print(d.X)

    d.reset_baselines()
    d.set_tolower(True)
    d.set_strip(True)
    d.set_one_baseline('gender', 'm')
    d.make_X()
    print(d.X)
    d.show_factor_info()

    d.reset_baselines()
    d.set_tolower(True)
    d.set_one_baseline('gender', 'X')
    d.make_X()
    d.show_factor_info()
