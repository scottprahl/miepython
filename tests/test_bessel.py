import unittest
import numpy as np
from miepython.bessel import *
from scipy.special import spherical_jn


class TestBesselFunctions(unittest.TestCase):
    def setUp(self):
        self.z = 1.5 - 0.5j
        self.tolerance = 1e-5

    def test_spherical_h1(self):
        # Expected values from WolframAlpha
        expected = [
            1.01008 + 0.258943j,
            0.813202 - 0.652698j,
            0.845303 - 0.945878j,
            2.66858 - 1.33963j,
        ]

        for n in range(4):
            result = spherical_h1(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_spherical_h2(self):
        # Expected values from WolframAlpha
        expected = [
            0.354426 + 0.146745j,
            0.0365618 + 0.513358j,
            -0.596630 + 0.799237j,
            -2.62569 + 1.28772j,
        ]

        for n in range(4):
            result = spherical_h2(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_riccati_bessel_jn(self):
        # Expected values from WolframAlpha
        expected = [
            1.1248 - 0.0368608j,
            0.602488 - 0.316946j,
            0.149845 - 0.172150j,
            0.0191954 - 0.0496582j,
        ]

        for n in range(4):
            result = riccati_bessel_jn(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_riccati_bessel_h1(self):
        # Expected values from WolframAlpha
        expected = [
            1.64459 - 0.116626j,
            0.893454 - 1.38565j,
            0.795015 - 1.841470j,
            3.33306 - 3.34374j,
        ]

        for n in range(4):
            result = riccati_bessel_h1(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_riccati_bessel_h2(self):
        # Expected values from WolframAlpha
        expected = [
            0.605011 + 0.0429043j,
            0.311522 + 0.751756j,
            -0.495326 + 1.497170j,
            -3.29467 + 3.24443j,
        ]

        for n in range(4):
            result = riccati_bessel_h2(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_spherical_jn(self):
        # Expected values from WolframAlpha
        expected = [
            -0.424882 + 0.0696702j,
            0.144527 + 0.1164950j,
            0.157083 - 0.0122946j,
            0.0520946 - 0.028186j,
        ]

        for n in range(0, 4):
            result = d_spherical_jn(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_spherical_h1(self):
        # Expected values from WolframAlpha
        expected = [
            -0.813202 + 0.652698j,
            -0.226842 + 0.71690j,
            -1.275870 + 0.542701j,
            -6.63101 + 0.134375j,
        ]

        for n in range(1, 4):
            result = d_spherical_h1(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_spherical_h2(self):
        # Expected values from WolframAlpha
        expected = [
            -0.0365618 - 0.513358j,
            0.515895 - 0.483909j,
            1.590040 - 0.567290j,
            6.7352 - 0.190747j,
        ]

        for n in range(1, 4):
            result = d_spherical_h2(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_riccati_bessel_jn(self):
        # Expected values from WolframAlpha
        expected = [
            0.0797651 + 0.51979j,
            0.699919 + 0.0328093j,
            0.353815 - 0.1703040j,
            0.0854978 - 0.0942821j,
        ]

        for n in range(4):
            result = d_riccati_bessel_jn(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_riccati_bessel_h1(self):
        # Expected values from WolframAlpha
        expected = [
            0.116626 + 1.64459j,
            0.831389 + 0.536072j,
            -0.797152 + 0.506108j,
            -7.21074 + 2.17743j,
        ]

        for n in range(1, 4):
            result = d_riccati_bessel_h1(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)

    def test_d_riccati_bessel_h2(self):
        # Expected values from WolframAlpha
        expected = [
            0.0429043 - 0.605011j,
            0.56845 - 0.470454j,
            1.50478 - 0.846717j,
            7.38174 - 2.366j,
        ]

        for n in range(1, 4):
            result = d_riccati_bessel_h2(n, self.z)
            self.assertAlmostEqual(result.real, expected[n].real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected[n].imag, delta=self.tolerance)


class TestAsymptotic(unittest.TestCase):
    def setUp(self):
        self.z = 1000
        self.tolerance = 1e-5

    #     def test_spherical_jn(self):
    #         for n in range(1, 4):
    #             expected = (-1j) ** n * np.exp(1j * self.z) / (1j * self.z)
    #             result = spherical_jn(n, self.z)
    #             self.assertAlmostEqual(result.real, expected.real, delta=self.tolerance)
    #             self.assertAlmostEqual(result.imag, expected.imag, delta=self.tolerance)

    def test_spherical_h1(self):
        expected = np.exp(1j * self.z) / (1j * self.z)
        for n in range(1, 4):
            expected = (-1j) ** n * np.exp(1j * self.z) / (1j * self.z)
            result = spherical_h1(n, self.z)
            self.assertAlmostEqual(result.real, expected.real, delta=self.tolerance)
            self.assertAlmostEqual(result.imag, expected.imag, delta=self.tolerance)


if __name__ == "__main__":
    unittest.main()
