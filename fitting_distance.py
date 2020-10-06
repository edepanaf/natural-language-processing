# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from vectorization_of_bag_collections.fitting_linear_vectorization import FittingLinearVectorization
from distance_on_bag_collections.cosine_distance import CosineDistance
# from distance_on_bag_collections.jensen_shannon_distance import JensenShannonDistance


DEFAULT_SPEED = 0.3
DEFAULT_NUMBER_OF_GRADIENT_STEPS = 6


class FittingDistance:

    def __init__(self, bag_collection, distance=CosineDistance(), item_to_weight=None):
        self.vectorize = FittingLinearVectorization(bag_collection, item_to_weight=item_to_weight)
        self.distance = distance

    def __call__(self, bags0, bags1):
        vectorization0 = self.vectorize(bags0)
        vectorization1 = self.vectorize(bags1)
        return self.distance(vectorization0, vectorization1)

    def fit(self, oracle_claims, speed=DEFAULT_SPEED, ratio_item_bag_fitting=0.5,
            number_of_gradient_steps=DEFAULT_NUMBER_OF_GRADIENT_STEPS):
        tuples_forms_arguments_intervals = [(self.distance, oracle_claim.pair_of_bags, oracle_claim.distance_interval)
                                            for oracle_claim in oracle_claims]
        self.vectorize.fit_from_tuples_of_forms_arguments_intervals(tuples_forms_arguments_intervals,
                                                                    speed=speed,
                                                                    ratio_item_bag_fitting=ratio_item_bag_fitting,
                                                                    number_of_gradient_steps=number_of_gradient_steps)

    def weight_from_item(self, item):
        if item not in self.vectorize.item_to_index:
            raise ValueError('Error in weight_from_item: ' + str(item) + ' does not appear in the collection.')
        return self.vectorize.item_weights_vector[self.vectorize.item_to_index[item]]

    def weight_from_bag(self, bag):
        return self.vectorize.bag_weights_vector[self.vectorize.bag_to_index[bag]]
