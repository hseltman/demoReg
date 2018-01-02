# -*- coding: utf-8 -*-
"""
Unit testing of class Reg
@author: hseltman
"""


import unittest
import pandas as pd
import numpy as np
# import io
# import sys
from Reg import Reg


class RegTestCase(unittest.TestCase):
    """Tests for `Reg.py`."""

    def setUp(self):
        self.dat = pd.DataFrame({'age': [25, 30, 35, 40],
                                 'male': ['m', 'M', 'f', 'F'],
                                 'score': [45, 52, 88, 51]})

    def tearDown(self):
        del self.dat

    def test_X_made_correctly(self):
        """Is X is a correctly sized array?"""
        r = Reg("score ~ age + male", self.dat)
        r.make_X()
        self.assertIsInstance(r.X, np.ndarray, msg='X not ndarray')
        self.assertTrue(all([a == b for (a, b) in zip(r.X.shape, (4, 3))]),
                        msg='X wrong size: {}'.format(r.X.shape))

    def test_fit_correct(self):
        """Does fit() generate correct output?"""
        r = Reg("score ~ age + male", self.dat)
        r.make_X()
        r.fit()
        self.assertIsInstance(r.bhat, pd.DataFrame,
                              msg="Attribute 'bhat' is not a DataFrame")
        self.assertTrue(r.bhat.shape == (3, 4),
                        msg="Shape of 'bhat' is {}".format(r.bhat.shape))
        self.assertTrue(all([a == b for (a, b) in
                            zip(r.bhat.columns, ('estimate', 'se', 't',
                                                 'p_value'))]),
                        msg="Columns of 'bhat' are {}".format(r.bhat.columns))
        self.assertTrue(all(round(r.bhat['estimate'], 1) ==
                            (182.0, -3.0, -51.0)),
                        msg="Estimates:\n{}".format(r.bhat['estimate']))
        self.assertTrue(all(round(r.bhat['se'], 1) ==
                            (165.7, 4.4, 49.2)),
                        msg="Std. errors:\n{}".format(r.bhat['se']))
        self.assertTrue(all(round(r.bhat['t'], 2) ==
                            (1.10, -0.68, -1.04)),
                        msg="t values:\n{}".format(r.bhat['t']))
        self.assertTrue(all(round(r.bhat['p_value'], 2) ==
                            (0.47, 0.62, 0.49)),
                        msg="p values:\n{}".format(r.bhat['p_value']))
        self.assertAlmostEqual(r.SSR, 484.0, msg="SSR = {}".format(r.SSR))
        self.assertEqual(r.df, 1, msg="df = {}".format(r.df))
        self.assertCountEqual(r.residual.round(2),
                              (-11.0, 11.0, 11.0, -11.0),
                              msg="residuals:\n{}".format(r.residual))
        self.assertAlmostEqual(r.se_residual, 22.0,
                               msg="se(res): {}".format(r.se_residual))

if __name__ == '__main__':
    unittest.main()
