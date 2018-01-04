# -*- coding: utf-8 -*-
"""
Unit testing of class Reg
@author: hseltman
"""


import unittest
import pandas as pd
import numpy as np
import io  # for StringIO
import sys  # for stdout
from DesignMatrix import DesignMat


class DesignMatTestCase(unittest.TestCase):
    """Tests for `Reg.py`."""

    def setUp(self):
        self.dat = pd.DataFrame({'age': [25, 30, 35, 40],
                                 'male': ['m', 'M', 'f ', 'F'],
                                 'tx': ['p', 'p', 'a', 'a'],
                                 'score': [45, 52, 88, 51]})

    def tearDown(self):
        del self.dat

    def test_DV_IVs(self):
        """Are DV and IV found?"""
        dm = DesignMat("score ~ age + male", self.dat)
        self.assertEqual(dm.DV, "score",
                         msg="DV is {} instead of 'score'".format(dm.DV))
        self.assertEqual(len(dm.IVs), 2, msg='len(IVs) != 2')
        self.assertTrue(all([a == b for (a, b) in
                             zip(dm.IVs, ('age', 'male'))]),
                        msg="IVs not 'age' and 'male'")

    def test_bad_DV(self):
        """Does a bad DV raise an exception?"""
        with self.assertRaisesRegex(Exception, "DV from 'formula'",
                                    msg="Did not detect bad DV"):
            DesignMat("scor ~ age + male", self.dat)

    def test_bad_IV(self):
        """Does a bad IV raise an exception?"""
        with self.assertRaisesRegex(Exception, "'gender' from",
                                    msg="Did not detect bad IV"):
            DesignMat("score ~ age + gender", self.dat)

    def test_X_made_correctly(self):
        """Is X is a correctly sized array?"""
        dm = DesignMat("score ~ age + male", self.dat)
        dm.make_X()
        self.assertIsInstance(dm.X, np.ndarray, msg='X not ndarray')
        self.assertTrue(all([a == b for (a, b) in zip(dm.X.shape, (4, 3))]),
                        msg='X wrong size: {}'.format(dm.X.shape))

    def test_show_output(self):
        """Test printed output of show_factor_info()"""
        capturedOutput = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = capturedOutput
        dm = DesignMat("score ~ age + male", self.dat)
        dm.make_X()
        dm.show_factor_info()
        sys.stdout = old_stdout
        out = capturedOutput.getvalue().split("\n")
        expect = ("Factor details:", "'male': baseline = 'F'",
                  "          others = 'M'", "")
        self.assertEqual(len(out), len(expect),
                         msg="output length != 3: {}".format(len(out)))
        self.assertTrue(all(o == e for (o, e) in zip(out, expect)),
                        msg="output is:\n{}".format("\n".join(out)))

    def test_no_factor_manipulation(self):
        """Are there five levels when factor manipulation is absent?"""
        dm = DesignMat("score ~ age + male", self.dat)
        dm.set_toupper(False)
        dm.make_X()
        self.assertEqual(dm.X.shape[1], 5,
                         msg="Need 5 columns with no casefolding for 'male'")

    def test_no_strip(self):
        """Are there four levels when stripping is absent?"""
        dm = DesignMat("score ~ age + male", self.dat)
        dm.set_strip(False)
        dm.set_toupper(True)
        dm.make_X()
        self.assertEqual(dm.X.shape[1], 4,
                         msg="Need 4 columns with no stripping")

    def test_lowercase(self):
        """Is baseline 'm' with lower case and custom baseline?"""
        dm = DesignMat("score ~ age + male", self.dat)
        dm.set_toupper(False)
        dm.set_tolower(True)
        dm.set_one_baseline('male', 'm')
        dm.make_X()
        self.assertEqual(dm.baselines['male'], 'm',
                         msg="Baseline for 'male' should be 'm'")

    def test_custom_baselines(self):
        """Is baseline 'm' with lower case and custom baseline?"""
        dm = DesignMat("score ~ age + male + tx", self.dat)
        dm.set_baselines({'male': 'M', 'tx': 'P'})
        dm.make_X()
        self.assertEqual(dm.baselines['tx'], 'P',
                         msg="Baseline for 'tx' should be 'P'")


if __name__ == '__main__':
    unittest.main()
