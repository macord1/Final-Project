import unittest
import Trial2
import numpy as np
'''
FINAL PROJECT

Molly Accord
Sreelakshmi Sunil

'''
'''
This file contains tests for some of the functions in Lazor.py
'''

class TestCases(unittest.TestCase):

    '''
    Performs Unit test on various functions in Laser class
    '''

    def PCA_analysis_test(self):
        '''
        Tests PCA_analysis_test function
        '''
        temp = [[1,2,4,5],[4,5,6,7],[5,6,7,8]]
        # the features size should be rx2
        # where are is no of rowes of passed matrix
        b = len(temp)
        a_1 = len(PCA_analysis(self,temp)[0]); b_1 =len(PCA_analysis(self,temp))
        self.assertEqual((a_1, b), (2, b_1))





if __name__ == '__main__':

    unittest.main()