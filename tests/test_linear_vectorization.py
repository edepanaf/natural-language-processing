# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from vectorization_of_bag_collections.linear_vectorization import *


bags = ['aa', 'ab', 'bbb']
item_to_weight = {'a': 1, 'b': 2}
bag_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
vectorization = LinearVectorization(bag_to_weight, item_to_weight)
bags0 = {'ab'}
bags1 = {'bbb'}


class TestLinearVectorization(unittest.TestCase):

    def test_log_of_ratio_zero_if_null_denominator(self):
        numerator = 5.
        denominator = 2.
        expected = np.log(numerator / denominator)
        computed = log_of_ratio_zero_if_null_denominator(numerator, denominator)
        self.assertAlmostEqual(expected, computed)
        self.assertAlmostEqual(log_of_ratio_zero_if_null_denominator(numerator, 0.), 0.)

    def test_call(self):
        vector0 = vectorization(bags0)
        vector1 = vectorization(bags1)
        self.assertTrue(are_almost_colinear_vectors(vector0, make_vector([2, 4])) or
                        are_almost_colinear_vectors(vector0, make_vector([4, 2])))
        self.assertTrue(are_almost_colinear_vectors(vector1, make_vector([0, 18])) or
                        are_almost_colinear_vectors(vector1, make_vector([18, 0])))
        self.assertTrue(are_almost_colinear_vectors(vector0 + vector1, vectorization({'ab', 'bbb'})))

    def test_tfidf_item_weights(self):
        tfidf_vectorization = LinearVectorization(bags)
        expected_item_weights_vector = make_vector([np.log(3 / 2), np.log(3 / 2)])
        expected_item_weights_vector /= sum(expected_item_weights_vector)
        self.assertTrue(are_almost_colinear_vectors(tfidf_vectorization.item_weights_vector,
                                                    expected_item_weights_vector))

    def test_input_dictionary_bag(self):
        vectorize = LinearVectorization(bag_to_weight)
        self.assertEqual(sum(vectorize.bag_weights_vector), 6.)


if __name__ == '__main__':
    unittest.main()
