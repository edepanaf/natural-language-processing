# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import math


def get_jensen_shannon_distance_and_variance(probabilities0, probabilities1):
    distribution = get_jensen_shannon_distribution(probabilities0, probabilities1)
    distance = math.sqrt(distribution.get_mean())
    variance = distribution.get_variance()
    return distance, variance


def get_jensen_shannon_distribution(probabilities0, probabilities1):
    mixture = mixture_from_probabilities(probabilities0, probabilities1)
    values0 = [information_from_probability(mixture[i]) - information_from_probability(probabilities0[i])
               for i in range(len(probabilities0))]
    values1 = [information_from_probability(mixture[i]) - information_from_probability(probabilities1[i])
               for i in range(len(probabilities1))]
    probabilities = list(map(lambda p: 0.5 * p, probabilities0)) + list(map(lambda p: 0.5 * p, probabilities1))
    return Distribution(values0 + values1, probabilities)


def mixture_from_probabilities(probabilities0, probabilities1):
    return [0.5 * probabilities0[index] + 0.5 * probabilities1[index] for index in range(len(probabilities0))]


def information_from_probability(p):
    if p == 0.:
        return 0.
    return - math.log(p, 2)


class Distribution:

    def __init__(self, values, probabilities):
        if len(values) != len(probabilities):
            raise ValueError
        self.values = values
        self.probabilities = probabilities
        self.mean = None

    def __len__(self):
        return len(self.values)

    def get_moment(self, order):
        return sum([self.probabilities[index] * self.values[index]**order for index in range(len(self))])

    def get_mean(self):
        if self.mean is None:
            self.mean = self.get_moment(1)
        return self.mean

    def get_variance(self):
        return self.get_moment(2) - self.get_mean()**2


def information_distribution_from_probabilities(probabilities):
    values = list(map(information_from_probability, probabilities))
    return Distribution(values, probabilities.copy())


def entropy_from_probabilities(probabilities):
    information_distribution = information_distribution_from_probabilities(probabilities)
    return information_distribution.get_mean()
