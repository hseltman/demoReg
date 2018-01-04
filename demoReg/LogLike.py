#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Univariate log-likelihood function

Given a data value, parameters, and a distribution,
return the log of the likelihood.

Currently implemented:
logLike(value, (mu, sd), "normal")
logLike(value, (n, p), "binomial")
"""

import scipy.stats as ss


def logLike(value, param=(0, 1), dist="normal"):
    if dist == "normal":
        return ss.norm.logpdf(value, loc=param[0], scale=param[1])
    elif dist == "binomial":
        return ss.binom.logpmf(value, n=param[0], p=param[1])
    else:
        raise(ValueError("currently 'dist' must be 'normal' or 'bimomial'"))


if __name__ == "__main__":
    import math
    print("test code for logLike:")
    print("likelihood of 5~N(3, var=2) =",
          math.exp(logLike(5, (3, 2))))
    print("likelihood of 2~B(5, p=0.5) =",
          math.exp(logLike(2, (5, 0.5), "binomial")))
