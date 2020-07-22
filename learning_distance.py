# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import random
from matrix_operations import *
from distance import Distance
from oracle_claim import OracleClaim

DEFAULT_NUMBER_OF_ITERATIONS = 5


class LearningDistance(Distance):

    def __init__(self, bags, item_to_weight=None, bag_to_weight=None):
        super().__init__(bags, item_to_weight, bag_to_weight)

    def learn(self, oracle_claims, ratio_item_bag_learning=0.5, speed=0.5,
              number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS):
        """ 'speed' is the fraction of the straight-line distance to the target
        traversed in one gradient descent step. """
        oracle_claims = list(oracle_claims)
        for _ in range(number_of_iterations):
            random.shuffle(oracle_claims)
            for oracle_claim in oracle_claims:
                self.apply_gradient_descent_step_for_one_oracle_claim(oracle_claim,
                                                                      ratio_item_bag_learning=ratio_item_bag_learning,
                                                                      speed=speed)

    def apply_gradient_descent_step_for_one_oracle_claim(self, oracle_claim, ratio_item_bag_learning=0.5, speed=1.):
        enriched_oracle_claim = EnrichedOracleClaim(oracle_claim, self, speed=speed)
        if enriched_oracle_claim.has_bad_values():
            return None
        rescaling_item_vector, rescaling_bag_vector = \
            self.compute_rescaling_vectors(enriched_oracle_claim, ratio_item_bag_learning)
        self.item_weights_vector = coefficient_wise_vector_product(rescaling_item_vector, self.item_weights_vector)
        self.bag_weights_vector = coefficient_wise_vector_product(rescaling_bag_vector,
                                                                  self.bag_weights_vector)

    def compute_rescaling_vectors(self, enriched_oracle_claim, ratio_item_bag_learning):
        gradient_item, gradient_bag = self.compute_item_and_bag_gradients(enriched_oracle_claim,
                                                                          ratio_item_bag_learning)
        rescaling_item_vector = rescale_vector_from_gradient_and_speed(gradient_item, enriched_oracle_claim.speed)
        rescaling_bag_vector = rescale_vector_from_gradient_and_speed(gradient_bag, enriched_oracle_claim.speed)
        return rescaling_item_vector, rescaling_bag_vector

    def compute_item_and_bag_gradients(self, enriched_oracle_claim, ratio_item_bag_learning):
        eoc = enriched_oracle_claim
        r = ratio_item_bag_learning
        w0 = eoc.argument0.vectorization / eoc.argument0.norm
        w1 = eoc.argument1.vectorization / eoc.argument1.norm
        w01 = w0 - (1. - eoc.current_distance) * w1
        w10 = w1 - (1. - eoc.current_distance) * w0
        partial_gradient_item = coefficient_wise_vector_product(w01, w1) + coefficient_wise_vector_product(w10, w0)
        partial_gradient_bag = (dot_dot_matrix_dot_products(eoc.argument1.vector,
                                                            self.bag_weights_vector,
                                                            transpose_matrix(self.item_bag_matrix),
                                                            self.item_weights_vector,
                                                            w01) / eoc.argument1.norm +
                                dot_dot_matrix_dot_products(eoc.argument0.vector,
                                                            self.bag_weights_vector,
                                                            transpose_matrix(self.item_bag_matrix),
                                                            self.item_weights_vector,
                                                            w10) / eoc.argument0.norm)
        common_factor = (eoc.current_distance - eoc.target_distance) / \
                        (r ** 2 * norm_from_vector(partial_gradient_item) ** 2 +
                         (1 - r) ** 2 * norm_from_vector(partial_gradient_bag) ** 2)
        gradient_item = common_factor * r ** 2 * partial_gradient_item
        gradient_bag = common_factor * (1 - r) ** 2 * partial_gradient_bag
        return gradient_item, gradient_bag


class EnrichedOracleClaim(OracleClaim, MemoryArgumentsVectorsVectorizationsNorms):

    def __init__(self, oracle_claim, distance_object, speed=1.):
        OracleClaim.__init__(self, oracle_claim.pair_of_bag_collections, oracle_claim.distance_interval)
        MemoryArgumentsVectorsVectorizationsNorms.__init__(self)
        self.speed = speed
        self.current_distance = distance_object(*self.pair_of_bag_collections, memory=self)
        self.target_distance = closest_point_from_interval(self.current_distance, self.distance_interval)
        self.target_distance = (self.current_distance + self.speed * (self.target_distance - self.current_distance))

    def has_bad_values(self):
        return (math.isclose(self.current_distance, self.target_distance)
                or math.isclose(self.argument0.norm, 0) or math.isclose(self.argument1.norm, 0))


def closest_point_from_interval(value, interval):
    lower_bound, upper_bound = interval
    if value < lower_bound:
        return lower_bound
    if value > upper_bound:
        return upper_bound
    return value


def rescale_vector_from_gradient_and_speed(gradient, speed):
    gradient = rescale_vector_to_satisfy_lower_negative_bound(gradient, -1. * speed)
    one_vector = one_vector_from_length(len(gradient))
    return one_vector + gradient


def dot_dot_matrix_dot_products(v0, v1, m, v2, v3):
    v23 = coefficient_wise_vector_product(v2, v3)
    mv23 = m * v23
    v01 = coefficient_wise_vector_product(v0, v1)
    v01mv23 = coefficient_wise_vector_product(v01, mv23)
    return v01mv23
