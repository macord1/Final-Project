import unittest
from Final import PCA_analysis
from Final import cluster_Kmeans
from Final import NEO_function

'''
FINAL PROJECT

Molly Accord
Sreelakshmi Sunil

'''
'''
This file contains tests for some of the functions in Final.py
'''


class TestCases(unittest.TestCase):

    '''
    Performs Unit test on various functions in Final.py
    '''

    def test_PCA_analysis(self):
        '''
        Tests PCA_analysis_test function
        '''
        temp =[[3.1234, 4.567, 1 ,2],[1, 2, 3, 4], [5, 6, 7, 8]]

        # the features size should be rx2
        # where are is no of rowes of passed matrix

        self.temp_size = len(temp)
       
        self.features = PCA_analysis(temp)
        a_1 = len(self.features[0])
        b_1 = len(self.features)
        
        self.assertEqual(a_1, 2)
        self.assertEqual(b_1, self.temp_size)

    def test_cluster_KMeans(self):
        '''
        Tests KMeans function
        '''
        temp =[[3.1234, 4.567, 1 ,2],[1, 2, 3, 4], [5, 6, 7, 8]]
        kmean = cluster_Kmeans(PCA_analysis(temp))
        self.assertEqual(len(kmean),len(temp))

        # checking for more than 3 clusters
        for i in kmean:
            if i not in {0,1,2}:
                print ("Error")
                break

    def test_NEO_function(self):
        '''
        Test NEO_function in Final
        '''
        bpf = [-5.7975308847776335, -5.756946956500628, -3.6020370537049997]
        b = len(bpf)
        self.assertEqual(len(NEO_function(bpf)), b)


if __name__ == '__main__':

    unittest.main()
