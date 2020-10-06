# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from vectorization_of_bag_collections.vector_space import *


bags = ['banana', 'ananas', 'base']
vector_space = VectorSpace(bags)
# item_to_index = {'b': 4, 'a': 3, 'n': 2, 's': 1, 'e': 0}
# bag_to_index = {'banana': 1, 'ananas': 2, 'base': 0}
# matrix = csc_matrix([[0, 0, 1],
#                      [0, 1, 1],
#                      [2, 2, 0],
#                      [3, 3, 1],
#                      [1, 0, 1]])


class TestVectorSpace(unittest.TestCase):

    def test_vector_length(self):
        vector = vector_space.bag_vector_from_collection(['ananas', 'banana'])
        projection = matrix_vector_product(vector_space.item_bag_matrix, vector)
        self.assertEqual(len(projection), len(vector_space.item_to_index))

    def test_count_bags_containing_item(self):
        self.assertEqual(vector_space.count_bags_containing_item('a'), 3)
        self.assertEqual(vector_space.count_bags_containing_item('b'), 2)
        self.assertEqual(vector_space.count_bags_containing_item('n'), 2)
        self.assertEqual(vector_space.count_bags_containing_item('e'), 1)
        self.assertEqual(vector_space.count_bags_containing_item('f'), 0)

    def test_map_to_index_from_iterable(self):
        bag = 'abacbde'
        computed = map_to_index_from_iterable(bag)
        computed_list = ['' for _ in range(len(bag))]
        for item, index in computed.items():
            computed_list[index] = item
        for letter in bag:
            self.assertIn(letter, computed_list)

    def test_union_from_iterables(self):
        string_list = ['', 'abc', 'de', '', 'f', '']
        computed = union_from_iterables(string_list)
        computed_string = ''
        for letter in computed:
            computed_string += letter
        self.assertEqual(computed_string, 'abcdef')

    def test_constant_distribution_from_collection(self):
        collection = {1, 2, 3}
        expected_distribution = {1:1., 2:1., 3:1.}
        computed_distribution = weight_one_dict_from_collection(collection)
        self.assertEqual(expected_distribution, computed_distribution)


if __name__ == '__main__':
    unittest.main()
