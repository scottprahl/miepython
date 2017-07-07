#! /usr/bin/env python

"""
Copyright 2017 Scott Prahl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import unittest
import numpy as np
from . import miepython
import sys

def run_tests():
    """Tests for the Mie code.

       Runs several tests that test the Mie code. All tests should return ok.
       If they don't, please contact the author.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(MieTests)
    unittest.TextTestRunner(verbosity=3).run(suite)

class MieTests(unittest.TestCase):

    def test_bh_dielectric(self):
        m = 1.55
        lambdaa = 0.6328
        radius = 0.525
        x = 2*np.pi*radius/lambdaa
        qext, qsca, qabs, qback, g = miepython.mie(m,x)

        self.assertAlmostEqual(qext, 3.10543, delta=0.00001)
        self.assertAlmostEqual(qsca, 3.10543, delta=0.00001)
        self.assertAlmostEqual(qabs, 0.00000, delta=0.00001)
        self.assertAlmostEqual(qback,2.92534, delta=0.00001)

    def test_non_dielectric(self):
        m = 1.55-0.1j
        lambdaa = 0.6328
        radius = 0.525
        x = 2*np.pi*radius/lambdaa
        qext, qsca, qabs, qback, g = miepython.mie(m,x)

        self.assertAlmostEqual(qext, 2.86165188243, delta=1e-7)
        self.assertAlmostEqual(qsca, 1.66424911991, delta=1e-7)
        self.assertAlmostEqual(qabs, 1.19740276252, delta=1e-7)
        self.assertAlmostEqual(qback,0.20599534080, delta=1e-7)
        self.assertAlmostEqual(g,    0.80128972639, delta=1e-7)

    def test_small_spheres(self):

        m = 0.75
        x = 0.099
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000007, delta=1e-6)
        self.assertAlmostEqual(g,    0.001448, delta=1e-6)
        x=0.101
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000008, delta=1e-6)
        self.assertAlmostEqual(g,    0.001507, delta=1e-6)

        m = 1.5 -1j
        x = 0.055
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.101491, delta=1e-6)
        self.assertAlmostEqual(g,    0.000491, delta=1e-6)
        x=0.056
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.103347, delta=1e-6)
        self.assertAlmostEqual(g,    0.000509, delta=1e-6)

        m = 1e-10 - 1e10j
        x=0.099;
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000321, delta=1e-6)
        self.assertAlmostEqual(g,   -0.397357, delta=1e-6)
        x=0.101;
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000348, delta=1e-6)
        self.assertAlmostEqual(g,   -0.397262, delta=1e-6)

        m = 0 - 1e10j
        x=0.099;
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000321, delta=1e-6)
        self.assertAlmostEqual(g,   -0.397357, delta=1e-6)
        x=0.101;
        qext, qsca, qabs, qback, g = miepython.mie(m,x)
        self.assertAlmostEqual(qext, 0.000348, delta=1e-6)
        self.assertAlmostEqual(g,   -0.397262, delta=1e-6)  

    def test_single_nonmagnetic(self):
        m = 1.5-0.5j
        x = 2.5
        qext, qsca, qabs, qback, g = miepython.mie(m,x)

        self.assertAlmostEqual(qext, 2.562873497454734, delta=1e-7)
        self.assertAlmostEqual(qsca, 1.097071819088392, delta=1e-7)
        self.assertAlmostEqual(qabs, 1.465801678366342, delta=1e-7)
        self.assertAlmostEqual(qback,0.123586468179818, delta=1e-7)
        self.assertAlmostEqual(g,    0.748905978948507, delta=1e-7)

#        S12_ref = (complex(-0.49958438416709694,-0.24032581667666403),
#                   complex(0.11666852712178288,0.051661382367147853))
#        S12 = mie.S12(-0.6)
#        self.assertLess(abs(S12_ref[0]-S12[0])/S12_ref[0], epsilon)
#        self.assertLess(abs(S12_ref[1]-S12[1])/S12_ref[1], epsilon)

    def test_magnetic(self):
        m = 1.6 - 1.4j
        x = 4.0
        qext, qsca, qabs, qback, g = miepython.mie(m,x)

        self.assertAlmostEqual(qext, 2.666558259429107, delta=1e-7)
        self.assertAlmostEqual(qsca, 1.125546094688389, delta=1e-7)
        self.assertAlmostEqual(qabs, 1.541012164740718, delta=1e-7)
        self.assertAlmostEqual(qback,0.007245317404096, delta=1e-7)
        self.assertAlmostEqual(g,    0.899559819378382, delta=1e-7)
                

#        S12_ref = (complex(0.14683196954000932,-0.017479181764394575),
#                   complex(-0.12475414168001844,0.28120475717321358))
#        S12 = mie.S12(-0.6)
#        self.assertLess(abs(S12_ref[0]-S12[0])/S12_ref[0], epsilon)
#        self.assertLess(abs(S12_ref[1]-S12[1])/S12_ref[1], epsilon)


#     def test_errors(self):
#         mie = Mie()
# 
#         #test that negative values of x fail
#         def test_x():
#             mie.x = -1.0
#         self.assertRaises(ValueError, test_x)
#         mie.x = 1.0
# 
#         #test that a missing m fails
#         self.assertRaises(ValueError, mie.qext)
#         mie.m = complex(1.5,0.5)
# 
#         #test that y<x fails (y==x is permitted)
#         def test_y():
#             mie.y = 0.5
#         self.assertRaises(ValueError, test_y)
#         mie.y = 1.5
# 
#         #test that setting y without m2 fails
#         self.assertRaises(ValueError, mie.qext)
#         mie.m2 = complex(1.2,0.5)
# 
#         #test that invalid values of u fail
#         self.assertRaises(ValueError, mie.S12, -1.5)
# 
#         mie.mu = complex(1.5,0.6)
#         #test that multilayered particles with mu fail
#         self.assertRaises(ValueError, mie.qext)


if __name__ == '__main__':
    unittest.main()
