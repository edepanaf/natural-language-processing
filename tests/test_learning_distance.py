# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from learning_distance import *
from oracle_claim import OracleClaim

bags = ['aa', 'ab', 'bbb']
item_to_weight = {'a': 1, 'b': 2}
bag_to_weight = {'aa': 1, 'ab': 2, 'bbb': 3}
distance = LearningDistance(bags, item_to_weight, bag_to_weight)
bags0 = {'ab'}
bags1 = {'bbb'}
bags2 = {'ab', 'ab'}
bags3 = {'bbb', 'aa'}


class TestLearningDistance(unittest.TestCase):

    def test_learn_correct_interval_target(self):
        bag0 = ('a', 'a', 'a', 'b')
        bag1 = ('a', 'a', 'a', 'b')
        bag2 = ('a', 'a', 'b', 'b')
        bag3 = ('a', 'c')
        all_bags = [bag0, bag1, bag2, bag3]
        item_weights = {'a': 1., 'b': 2., 'c': 0.5}
        bag_weights = {bag0: 1., bag1: 1., bag2: 1., bag3: 1.}
        interval_true_distance = (0.1, 0.2)
        oracle_claim = OracleClaim(({bag0, bag1}, {bag1, bag3}), interval_true_distance)
        learning_distance = LearningDistance(all_bags, item_to_weight=item_weights,
                                             bag_to_weight=bag_weights)
        # old_distance = learning_distance(oracle_claim.bags_pair[0], oracle_claim.bags_pair[1])
        learning_distance.learn({oracle_claim}, ratio_item_bag_learning=1., number_of_iterations=5,
                                speed=0.5)
        new_distance = learning_distance(oracle_claim.pair_of_bag_collections[0],
                                         oracle_claim.pair_of_bag_collections[1])
        self.assertTrue(abs(new_distance - 0.1) < abs(new_distance - 0.2))

    def test_learn(self):
        current_distance0 = distance(bags0, bags1)
        target_distance0 = current_distance0 * 2.
        oracle_claim0 = OracleClaim((bags0, bags1), (target_distance0, 1.))
        current_distance1 = distance(bags2, bags3)
        target_distance1 = current_distance1 * 2.
        oracle_claim1 = OracleClaim((bags2, bags3), (target_distance1, 1.))
        distance.learn([oracle_claim0, oracle_claim1], number_of_iterations=5)
        obtained_distance0 = distance(bags0, bags1)
        obtained_distance1 = distance(bags2, bags3)
        self.assertTrue(abs(obtained_distance0 - target_distance0) < abs(current_distance0 - target_distance0))
        self.assertTrue(abs(obtained_distance1 - target_distance1) < abs(current_distance1 - target_distance1))

    def test_apply_gradient_descent_step_for_one_oracle_claim_larger_distance(self):
        current_distance = distance(bags0, bags1)
        target_distance = current_distance * 2.
        oracle_claim = OracleClaim((bags0, bags1), (target_distance, 1.))
        distance.apply_gradient_descent_step_for_one_oracle_claim(oracle_claim, speed=0.5)
        obtained_distance = distance(bags0, bags1)
        self.assertTrue(abs(obtained_distance - target_distance) < abs(current_distance - target_distance))

    def test_apply_gradient_descent_step_for_one_oracle_claim_smaller_distance(self):
        current_distance = distance(bags0, bags1)
        target_distance = current_distance / 2.
        oracle_claim = OracleClaim((bags0, bags1), (0., target_distance))
        distance.apply_gradient_descent_step_for_one_oracle_claim(oracle_claim, speed=0.5)
        obtained_distance = distance(bags0, bags1)
        self.assertTrue(abs(obtained_distance - target_distance) < abs(current_distance - target_distance))

    def test_compute_item_and_bag_gradients(self):
        current_distance = distance(bags0, bags1)
        target_distance = current_distance * 2.
        oracle_claim = OracleClaim((bags0, bags1), (target_distance, 1.))
        enriched_oracle_claim = EnrichedOracleClaim(oracle_claim, distance, speed=0.5)
        _, bag_gradient = distance.compute_item_and_bag_gradients(enriched_oracle_claim, 1.)
        self.assertTrue(is_almost_zero_vector(bag_gradient))
        item_gradient, _ = distance.compute_item_and_bag_gradients(enriched_oracle_claim, 0.)
        self.assertTrue(is_almost_zero_vector(item_gradient))

    def test_closest_point_from_interval(self):
        interval = (-2, 4)
        self.assertEqual(closest_point_from_interval(-3, interval), -2)
        self.assertEqual(closest_point_from_interval(-2, interval), -2)
        self.assertEqual(closest_point_from_interval(5, interval), 4)
        self.assertEqual(closest_point_from_interval(1, interval), 1)

    def test_rescale_vector_from_gradient_and_speed(self):
        vector = create_vector([6, 4, -2, 0])
        computed = rescale_vector_from_gradient_and_speed(vector, 1.)
        expected = create_vector([1 + 6/2, 1 + 4/2, 1 + -2/2, 1])
        self.assertTrue(are_equal_vectors(expected, computed))
        vector = create_vector([0, 1, 2, 3])
        computed = rescale_vector_from_gradient_and_speed(vector, 1.)
        expected = create_vector([1, 2, 3, 4])
        self.assertTrue(are_equal_vectors(expected, computed))


def random_vector(length):
    return create_vector([0.5 - random.random() for _ in range(length)])


if __name__ == '__main__':
    unittest.main()
