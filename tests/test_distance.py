# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from distance import *

bags = ['aa', 'ab', 'bbb']
item_to_weight = {'a': 1, 'b': 2}
bag_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
distance_object = Distance(bags, item_to_weight, bag_to_weight)
bags0 = {'ab'}
bags1 = {'bbb'}


class MyTestCase(unittest.TestCase):

    def test_vectorize(self):
        vector0 = distance_object.vectorize(bags0)
        vector1 = distance_object.vectorize(bags1)
        self.assertTrue(are_almost_colinear_vectors(vector0, create_vector([2, 4])) or
                        are_almost_colinear_vectors(vector0, create_vector([4, 2])))
        self.assertTrue(are_almost_colinear_vectors(vector1, create_vector([0, 18])) or
                        are_almost_colinear_vectors(vector1, create_vector([18, 0])))
        self.assertTrue(are_almost_colinear_vectors(vector0 + vector1, distance_object.vectorize({'ab', 'bbb'})))

    def test_tfidf(self):
        tfidf_distance = Distance(bags)
        expected_item_weights_vector = create_vector([math.log(3/2), math.log(3/2)])
        expected_item_weights_vector /= sum(expected_item_weights_vector)
        self.assertTrue(are_equal_vectors(tfidf_distance.item_weights_vector, expected_item_weights_vector))

    def test_get_jensen_shannon_distance_and_variance(self):
        distance, variance = distance_object.get_jensen_shannon_distance_and_variance(bags0, bags1)
        self.assertTrue(distance > 0.)
        self.assertTrue(variance > 0.)


if __name__ == '__main__':
    unittest.main()
