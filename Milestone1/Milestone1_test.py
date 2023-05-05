import unittest
import numpy as np
from Milestone1 import Simulation


class TestStreaming(unittest.TestCase):

    def test_mass_conservation(self):
        sim = Simulation()
        # Mass beforehand
        m1 = sim.f.sum(axis=(1, 2))
        # shift in every direction possible and test if mass is conserved
        for i in range(9):
            sim.streaming(i)
            m2 = sim.f.sum(axis=(1, 2))
            self.assertTrue(np.allclose(m1, m2))

    def test_shifting(self):
        sim = Simulation()
        sim.f = np.arange(15).reshape(1, 5, 3)
        top_right_shift = np.array([[[5, 3, 4],
                                     [8, 6, 7],
                                     [11, 9, 10],
                                     [14, 12, 13],
                                     [2, 0, 1]]])
        sim.streaming(5)        # 5 = top right shift
        np.all(sim.f == top_right_shift)


if __name__ == '__main__':
    unittest.main()
