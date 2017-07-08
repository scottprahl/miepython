#! /usr/bin/env python3

"""
Copyright 2017 Scott Prahl

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import unittest
import numpy as np
import miepython

def run_tests():
    """Tests for the mie code.

       Runs several tests that test the mie code. All tests should return ok.
       If they don't, please contact the author.
    """
    suite = unittest.TestLoader().loadTestsFromTestCase(low_level)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(non_absorbing)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(absorbing)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(small)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(conducting)
    unittest.TextTestRunner().run(suite)
    suite = unittest.TestLoader().loadTestsFromTestCase(angle_scattering)
    unittest.TextTestRunner().run(suite)


#class interface(unittest.TestCase):

#     def test_errors(self):
#         mie = mie()
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

class low_level(unittest.TestCase):

	def test_01_log_derivatives(self):
		m = 1.0
		x = 1.0
		nstop = 10
		dn = miepython.Dn_calc(m,x,nstop)
		self.assertAlmostEqual(dn[9].real, 9.95228198, delta=0.00001)

		x = 62
		m = 1.28 - 1.37j
		nstop = 50
		dn = miepython.Dn_calc(m,x,nstop)
		self.assertAlmostEqual(dn[10].real, 0.004087, delta=0.00001)
		self.assertAlmostEqual(dn[10].imag, 1.0002620, delta=0.00001)

	def test_02_an_bn(self):
		m = 4.0/3.0
		x = 50
		a, b = miepython.mie_An_Bn(m,x)
		print(a)
	#        self.assertAlmostEqual(a[0].real, 0.5311058892948411929, delta=0.00000001)
	#        self.assertAlmostEqual(a[1].imag,-0.4990314856310943073, delta=0.00000001)
	#        self.assertAlmostEqual(b[1].real, 0.093412567968, delta=0.00001)
	#        self.assertAlmostEqual(b[1].imag,-0.067160541299, delta=0.00001)

		m = 1.5-1.1j
		x = 2
		a, b = miepython.mie_An_Bn(m,x)
		self.assertAlmostEqual(a[0].real, 0.555091767665, delta=0.00001)
		self.assertAlmostEqual(a[0].imag, 0.158587776121, delta=0.00001)
		self.assertAlmostEqual(a[1].real, 0.386759705234, delta=0.00001)
		self.assertAlmostEqual(a[1].imag, 0.076275273072, delta=0.00001)
		self.assertAlmostEqual(b[1].real, 0.093412567968, delta=0.00001)
		self.assertAlmostEqual(b[1].imag,-0.067160541299, delta=0.00001)

		m = 1.1-25j
		x = 2
		a, b = miepython.mie_An_Bn(m,x)
		self.assertAlmostEqual(a[1].real, 0.324433578437, delta=0.0001)
		self.assertAlmostEqual(a[1].imag, 0.465627763266, delta=0.0001)
		self.assertAlmostEqual(b[1].real, 0.060464399088, delta=0.0001)
		self.assertAlmostEqual(b[1].imag,-0.236805417045, delta=0.0001)


class non_absorbing(unittest.TestCase):

	def test_03_bh_dielectric(self):
		m = 1.55
		lambdaa = 0.6328
		radius = 0.525
		x = 2*np.pi*radius/lambdaa
		qext, qsca, qabs, qback, g = miepython.mie(m,x)

		self.assertAlmostEqual(qext, 3.10543, delta=0.00001)
		self.assertAlmostEqual(qsca, 3.10543, delta=0.00001)
		self.assertAlmostEqual(qabs, 0.00000, delta=0.00001)
		self.assertAlmostEqual(qback,2.92534, delta=0.00001)
		self.assertAlmostEqual(g    ,0.63314, delta=0.00001)

	def test_05_wiscombe_non_absorbing(self):
		m=complex(0.75, 0.0);
		x=0.099;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000007, delta=1e-6)
		self.assertAlmostEqual(g,    0.001448, delta=1e-6)

		m=complex(0.75, 0.0);
		x=0.101;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000008, delta=1e-6)
		self.assertAlmostEqual(g,    0.001507, delta=1e-6)

		m=complex(0.75, 0.0);
		x=10.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 2.232265, delta=1e-6)
		self.assertAlmostEqual(g,    0.896473, delta=1e-6)

		m=complex(0.75, 0.0);
		x=1000.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 1.997908, delta=1e-6)
		self.assertAlmostEqual(g,    0.844944, delta=1e-6)

	def test_04_non_dielectric(self):
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

class absorbing(unittest.TestCase):
	def test_06_wiscombe_water_absorbing(self):
		m=complex(1.33, -0.00001);
		x=1.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.093923, delta=1e-6)
		self.assertAlmostEqual(g,    0.184517, delta=1e-6)

		x=100.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 2.096594, delta=1e-6)
		self.assertAlmostEqual(g,    0.868959, delta=1e-6)

		x=10000.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 1.723857, delta=1e-6)
		self.assertAlmostEqual(g,    0.907840, delta=1e-6)

	def test_07_wiscombe_absorbing(self):
		m=complex(1.5, -1.00);
		x=0.055;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000011, delta=1e-6)
		self.assertAlmostEqual(g,    0.000491, delta=1e-6)

		x=0.056;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000012, delta=1e-6)
		self.assertAlmostEqual(g,    0.000509, delta=1e-6)

		x=1.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.6634538, delta=1e-6)
		self.assertAlmostEqual(g,    0.192136, delta=1e-6)

		x=100.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 1.283697, delta=1e-3)
		self.assertAlmostEqual(g,    0.850252, delta=1e-3)

	#        x=10000.0;
	#        qext, qsca, qabs, qback, g = miepython.mie(m,x)
	#        self.assertAlmostEqual(qsca, 1.236574, delta=1e-6)
	#        self.assertAlmostEqual(g,    0.846310, delta=1e-6)

	def test_08_wiscombe_more_absorbing(self):
		m=complex(10.0, -10.00);
		x=1.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 2.049405, delta=1e-6)
		self.assertAlmostEqual(g,   -0.110664, delta=1e-6)

		x=100.0;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 1.836785, delta=1e-6)
		self.assertAlmostEqual(g,    0.556215, delta=1e-6)

	#         x=10000.0;
	#         qext, qsca, qabs, qback, g = miepython.mie(m,x)
	#         self.assertAlmostEqual(qsca, 1.795393, delta=1e-6)
	#         self.assertAlmostEqual(g,    0.548194, delta=1e-6)

	def test_09_single_nonmagnetic(self):
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

	#     def test_12_magnetic(self):
	#         m = 1.6 - 1.4j
	#         x = 4.0
	#         qext, qsca, qabs, qback, g = miepython.mie(m,x)
	# 
	#         self.assertAlmostEqual(qext, 2.666558259429107, delta=1e-7)
	#         self.assertAlmostEqual(qsca, 1.125546094688389, delta=1e-7)
	#         self.assertAlmostEqual(qabs, 1.541012164740718, delta=1e-7)
	#         self.assertAlmostEqual(qback,0.007245317404096, delta=1e-7)
	#         self.assertAlmostEqual(g,    0.899559819378382, delta=1e-7)
			

	#        S12_ref = (complex(0.14683196954000932,-0.017479181764394575),
	#                   complex(-0.12475414168001844,0.28120475717321358))
	#        S12 = mie.S12(-0.6)
	#        self.assertLess(abs(S12_ref[0]-S12[0])/S12_ref[0], epsilon)
	#        self.assertLess(abs(S12_ref[1]-S12[1])/S12_ref[1], epsilon)

class conducting(unittest.TestCase):

	def test_11_wiscombe_perfectly_conducting(self):
		m=complex(0.0, 0.0);
		x=0.099;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000321, delta=1e-4)
		self.assertAlmostEqual(g,   -0.397357, delta=1e-4)

		x=0.101;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 0.000348, delta=1e-6)
		self.assertAlmostEqual(g,   -0.397262, delta=1e-6)

		x=100;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 2.008102, delta=1e-6)
		self.assertAlmostEqual(g,    0.500926, delta=1e-6)

		x=10000;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qsca, 2.000289, delta=1e-6)
		self.assertAlmostEqual(g,    0.500070, delta=1e-6)

class small(unittest.TestCase):

	def test_10_small_spheres(self):
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

	#       m = 1e-10 - 1e10j
	#       x=0.099;
	#       qext, qsca, qabs, qback, g = miepython.mie(m,x)
	#       self.assertAlmostEqual(qext, 0.000321, delta=1e-6)
	#       self.assertAlmostEqual(g,   -0.397357, delta=1e-6)
	#       x=0.101;
	#       qext, qsca, qabs, qback, g = miepython.mie(m,x)
	#       self.assertAlmostEqual(qext, 0.000348, delta=1e-6)
	#       self.assertAlmostEqual(g,   -0.397262, delta=1e-6)

		m = 0 - 1e10j
		x=0.099;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qext, 0.000321, delta=1e-5)
		self.assertAlmostEqual(g,   -0.397357, delta=1e-4)
		x=0.101;
		qext, qsca, qabs, qback, g = miepython.mie(m,x)
		self.assertAlmostEqual(qext, 0.000348, delta=1e-5)
		self.assertAlmostEqual(g,   -0.397262, delta=1e-4)  

class angle_scattering(unittest.TestCase):

	def test_12_scatter_function(self):
		x=1.0;
		m=1.5-1.0j
		theta = np.arange(0,181,30)
		mu = np.cos(theta * np.pi/180)

		S1, S2 = miepython.mie_S1_S2(m,x,mu)

		self.assertAlmostEqual(S1[0].real, 0.584080, delta=1e-6)
		self.assertAlmostEqual(S1[0].imag, 0.190515, delta=1e-6)
		self.assertAlmostEqual(S2[0].real, 0.584080, delta=1e-6)
		self.assertAlmostEqual(S2[0].imag, 0.190515, delta=1e-6)

		self.assertAlmostEqual(S1[1].real, 0.565702, delta=1e-6)
		self.assertAlmostEqual(S1[1].imag, 0.187200, delta=1e-6)
		self.assertAlmostEqual(S2[1].real, 0.500161, delta=1e-6)
		self.assertAlmostEqual(S2[1].imag, 0.145611, delta=1e-6)

		self.assertAlmostEqual(S1[2].real, 0.517525, delta=1e-6)
		self.assertAlmostEqual(S1[2].imag, 0.178443, delta=1e-6)
		self.assertAlmostEqual(S2[2].real, 0.287964, delta=1e-6)
		self.assertAlmostEqual(S2[2].imag, 0.041054, delta=1e-6)

		self.assertAlmostEqual(S1[3].real, 0.456340, delta=1e-6)
		self.assertAlmostEqual(S1[3].imag, 0.167167, delta=1e-6)
		self.assertAlmostEqual(S2[3].real, 0.0362285, delta=1e-6)
		self.assertAlmostEqual(S2[3].imag,-0.0618265, delta=1e-6)

		self.assertAlmostEqual(S1[4].real, 0.400212, delta=1e-6)
		self.assertAlmostEqual(S1[4].imag, 0.156643, delta=1e-6)
		self.assertAlmostEqual(S2[4].real,-0.174875, delta=1e-6)
		self.assertAlmostEqual(S2[4].imag,-0.122959, delta=1e-6)

		self.assertAlmostEqual(S1[5].real, 0.362157, delta=1e-6)
		self.assertAlmostEqual(S1[5].imag, 0.149391, delta=1e-6)
		self.assertAlmostEqual(S2[5].real,-0.305682, delta=1e-6)
		self.assertAlmostEqual(S2[5].imag,-0.143846, delta=1e-6)

		self.assertAlmostEqual(S1[6].real, 0.348844, delta=1e-6)
		self.assertAlmostEqual(S1[6].imag, 0.146829, delta=1e-6)
		self.assertAlmostEqual(S2[6].real,-0.348844, delta=1e-6)
		self.assertAlmostEqual(S2[6].imag,-0.146829, delta=1e-6)

if __name__ == '__main__':
	unittest.main()
