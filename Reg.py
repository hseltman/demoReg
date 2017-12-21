#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 09:13:51 2017

@author: hseltman
"""

import pandas as pd
import logLike


class Reg:
    def __init__(self, formula, data):
        if not isinstance(formula, str):
            raise(TypeError("'formula' must be a string"))
        if not isinstance(data, pd.core.frame.DataFrame):
            raise(TypeError("'data' must be a pandas DataFrame"))
        self.formula = formula
        self.data = data

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

    def ll(self):
        return logLike.logLike(0)


if __name__ == "__main__":
    dat = pd.DataFrame({'age': [25, 30, 35, 40],
                        'male': [1, 1, 0, 0],
                        'score': [45, 52, 88, 51]})
    r = Reg("score ~ age + male", dat)
    print(r.ll())
    print(r.formula)
    print(r.data)
    print(r)
    repr(r)
