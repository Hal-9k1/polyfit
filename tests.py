from unittest import TestCase
from fit import fit, poly_eval

class PolyEval(TestCase):
    def test_zero_coeffs(self):
        self.assertEqual(poly_eval((0, 0, 0), 2), 0)

    def test_works(self):
        self.assertEqual(poly_eval((2000, -0.5, 0, 1, -2.75), 4.8), 648.3776, places=4)

class Fit(TestCase):
    def test_fit(self):
        
