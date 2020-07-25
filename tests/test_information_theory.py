# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import unittest
from information_theory import *
import random


coin_probabilities = [0.5, 0.5]


class TestInformationTheory(unittest.TestCase):

    def test_get_jensen_shannon_distance(self):
        probabilities0 = probabilities_from_list(random_list_from_length(5))
        probabilities1 = probabilities_from_list(random_list_from_length(5))
        distance, _ = get_jensen_shannon_distance_and_variance(probabilities0, probabilities1)
        mixture = mixture_from_probabilities(probabilities0, probabilities1)
        entropy0 = entropy_from_probabilities(probabilities0)
        entropy1 = entropy_from_probabilities(probabilities1)
        entropy_mixture = entropy_from_probabilities(mixture)
        self.assertAlmostEqual(distance, math.sqrt(entropy_mixture - 0.5 * (entropy0 + entropy1)))

    def test_mixture_from_probabilities(self):
        distribution = [0.25, 0.75]
        mixture = mixture_from_probabilities(coin_probabilities, distribution)
        self.assertAlmostEqual(mixture[0], 0.5 * (distribution[0] + coin_probabilities[0]))

    def test_information_from_probability(self):
        self.assertAlmostEqual(information_from_probability(0.5), 1.)
        self.assertAlmostEqual(information_from_probability(0.), 0.)

    def test_probabilities_from_list(self):
        my_list = random_list_from_length(2)
        probabilities = probabilities_from_list(my_list)
        self.assertAlmostEqual(sum(probabilities), 1.)
        self.assertAlmostEqual(my_list[0] / sum(my_list), probabilities[0])

    def test_entropy_from_probabilities(self):
        self.assertAlmostEqual(entropy_from_probabilities(coin_probabilities), 1.)
        probabilities0 = probabilities_from_list(random_list_from_length(4))
        probabilities1 = probabilities0 + [0.]
        self.assertAlmostEqual(entropy_from_probabilities(probabilities0), entropy_from_probabilities(probabilities1))

    def test_information_distribution_from_probabilities(self):
        distribution = information_distribution_from_probabilities(coin_probabilities)
        self.assertAlmostEqual(distribution.values[0], 1.)
        self.assertAlmostEqual(distribution.values[1], 1.)


if __name__ == '__main__':
    unittest.main()


def random_list_from_length(length):
    return [random.random() for _ in range(length)]