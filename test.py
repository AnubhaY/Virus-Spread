import unittest
import virusSpread

def timer(data, algo='matrix'):
    matrix = virusSpread.MatrixGenerator.generate(data, algo=algo)
    return matrix.timer_till_all_infected()


class TestInfection(unittest.TestCase):
    def test_basic_yes_m(self):
        data = [[2, 1, 0, 2, 1], [1, 1, 1, 1, 1], [1, 0, 0, 2, 1]]
        self.assertEqual(timer(data, 'matrix'), 2, "Should be 2")


    def test_basic_no_m(self):
        data = [[2, 1, 0, 2, 1], [0, 1, 1, 1, 1], [1, 0, 0, 2, 1]]
        self.assertEqual(timer(data, 'matrix'), -1, "Should be -1 for NO")

    def test_none_2_m(self):
        data = [[2]]
        self.assertEqual(timer(data, 'matrix'), 0, "Everyone is already infected")

    def test_none_1_m(self):
        data = [[1]]
        self.assertEqual(timer(data, 'matrix'), -1, "No infected present")

    def test_none_0_m(self):
        data = [[0]]
        self.assertEqual(timer(data, 'matrix'), -1, "No infected present")

    def test_none_m(self):
        data = [[]]
        self.assertEqual(timer(data, 'matrix'), -1, "No infected present")


    def test_basic_yes_g(self):
        data = [[2, 1, 0, 2, 1], [1, 1, 1, 1, 1], [1, 0, 0, 2, 1]]
        self.assertEqual(timer(data, 'grid'), 2, "Should be 2")


    def test_basic_no_g(self):
        data = [[2, 1, 0, 2, 1], [0, 1, 1, 1, 1], [1, 0, 0, 2, 1]]
        self.assertEqual(timer(data, 'grid'), -1, "Should be -1 for NO")

    def test_none_2_g(self):
        data = [[2]]
        self.assertEqual(timer(data, 'grid'), 0, "Everyone is already infected")

    def test_none_1_g(self):
        data = [[1]]
        self.assertEqual(timer(data, 'grid'), -1, "No infected present")

    def test_none_0_g(self):
        data = [[0]]
        self.assertEqual(timer(data, 'grid'), -1, "No infected present")

    def test_none_g(self):
        data = [[]]
        self.assertEqual(timer(data, 'grid'), -1, "No infected present")


    def test_big_m(self):
        data = [[1]*1000]*1000
        data[2][2] = 2
        self.assertEqual(timer(data, 'matrix'), 997, "Should be 997")
        
    def test_med_g(self):
        data = [[1]*100]*100
        data[2][2] = 2
        self.assertEqual(timer(data, 'grid'), 97, "Should be 97") 




if __name__=='__main__':
    unittest.main()
