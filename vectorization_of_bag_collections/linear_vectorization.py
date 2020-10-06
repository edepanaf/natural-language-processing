# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from matrix_operations import *
from vectorization_of_bag_collections.vector_space import VectorSpace


class LinearVectorization(VectorSpace):

    def __init__(self, bags, item_to_weight=None):
        super().__init__(bags)
        #
        self.item_weights_vector = None
        if item_to_weight is None:
            item_to_weight = self.tfidf_item_weights()
        self.set_item_weights(item_to_weight)
        #
        self.bag_weights_vector = None
        if isinstance(bags, dict):
            bag_to_weight = bags
        else:
            bag_to_weight = {bag: 1 / len(bag) for bag in self.bag_to_index}
        self.set_bag_weights(bag_to_weight)

    def __call__(self, bags):
        vector = self.bag_vector_from_collection(bags)
        return dot_matrix_dot_products(self.item_weights_vector, self.item_bag_matrix,
                                       self.bag_weights_vector, vector)

    def set_item_weights(self, item_to_weight):
        self.item_weights_vector = self.item_vector_from_dict(item_to_weight)

    def set_bag_weights(self, bag_to_weight):
        self.bag_weights_vector = self.bag_vector_from_dict(bag_to_weight)

    def get_item_weights(self):
        return self.item_dict_from_vector(self.item_weights_vector)

    def get_bag_weights(self):
        return self.bag_dict_from_vector(self.bag_weights_vector)

    def tfidf_item_weights(self):
        number_of_bags = len(self.bag_to_index)
        # Use of log_of_ratio_zero_if_null_denominator to handle the case
        # where the only bag containing an item has been removed (operation currently not supported).
        return {item: log_of_ratio_zero_if_null_denominator(number_of_bags, self.count_bags_containing_item(item))
                for item in self.item_to_index}


def log_of_ratio_zero_if_null_denominator(numerator, denominator):
    if denominator == 0:
        return 0.
    return np.log(numerator / denominator)
