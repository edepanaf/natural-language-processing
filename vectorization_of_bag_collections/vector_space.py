# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from matrix_operations import *


class VectorSpace:

    def __init__(self, bags):
        self.item_to_index = map_to_index_from_iterable(union_from_iterables(bags))
        self.bag_to_index = map_to_index_from_iterable(bags)
        self.item_bag_matrix = matrix_from_bags_and_index_maps(
            bags, self.item_to_index, self.bag_to_index)

    def item_vector_from_dict(self, item_distribution):
        return vector_from_index_and_value_maps(self.item_to_index, item_distribution)

    def bag_vector_from_dict(self, distribution_on_bags):
        return vector_from_index_and_value_maps(self.bag_to_index, distribution_on_bags)

    def item_dict_from_vector(self, item_vector):
        return dict_from_index_map_and_vector(self.item_to_index, item_vector)

    def bag_dict_from_vector(self, bag_vector):
        return dict_from_index_map_and_vector(self.bag_to_index, bag_vector)

    def bag_vector_from_collection(self, bags):
        distribution_on_bags = weight_one_dict_from_collection(bags)
        return self.bag_vector_from_dict(distribution_on_bags)

    def count_bags_containing_item(self, item):
        if item not in self.item_to_index:
            return 0
        return count_nonzero_entries_in_matrix_row(self.item_bag_matrix, self.item_to_index[item])


def map_to_index_from_iterable(iterable):
    map_to_index = dict()
    index = 0
    for item in iterable:
        if item not in map_to_index:
            map_to_index[item] = index
            index += 1
    return map_to_index


def union_from_iterables(iterables):
    for iterable in iterables:
        for item in iterable:
            yield item


def weight_one_dict_from_collection(collection):
    return {element: 1. for element in collection}
