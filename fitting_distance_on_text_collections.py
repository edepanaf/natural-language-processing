# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from fitting_distance import FittingDistance
from distance_on_bag_collections.cosine_distance import CosineDistance
from oracle_claim import OracleClaim


DEFAULT_MAX_FACTOR_LENGTH = 5


class FittingDistanceOnTextCollections:

    def __init__(self, text_collection, distance=CosineDistance(),
                 max_factor_length=DEFAULT_MAX_FACTOR_LENGTH,
                 factor_to_weight=None):
        self.text_to_bag = {text: bag_of_factors_from_text(clean_text(text), max_factor_length)
                            for text in text_collection}
        if isinstance(text_collection, dict):
            bags = {self.text_to_bag[text]: weight for text, weight in text_collection.items()}
        else:
            bags = self.text_to_bag.values()
        self.fitting_distance = FittingDistance(bags, distance, item_to_weight=factor_to_weight)

    def __call__(self, text_collection0, text_collection1):
        bags0 = self.bag_collection_from_text_collection(text_collection0)
        bags1 = self.bag_collection_from_text_collection(text_collection1)
        return self.fitting_distance(bags0, bags1)

    def fit(self, text_oracle_claims):
        bag_oracle_claims = {self.bag_oracle_claim_from_text_oracle_claim(claim) for claim in text_oracle_claims}
        self.fitting_distance.fit(bag_oracle_claims)

    def bag_collection_from_text_collection(self, text_collection):
        return {self.text_to_bag[text] for text in text_collection}

    def bag_oracle_claim_from_text_oracle_claim(self, text_oracle_claim):
        text_collection0, text_collection1 = text_oracle_claim.pair_of_bags
        bags0 = self.bag_collection_from_text_collection(text_collection0)
        bags1 = self.bag_collection_from_text_collection(text_collection1)
        return OracleClaim((bags0, bags1), text_oracle_claim.distance_interval)

    def weight_from_factor(self, factor):
        try:
            return self.fitting_distance.weight_from_item(factor)
        except ValueError:
            return 0.

    def weight_from_text(self, text):
        return self.fitting_distance.weight_from_bag(self.text_to_bag[text])


def bag_of_factors_from_text(text, max_factor_length=None):
    if max_factor_length is None:
        max_factor_length = len(text)
    return tuple(text[start:end] for start in range(len(text))
                 for end in range(start + 1, 1 + min(len(text), start + max_factor_length)))


def clean_text(text):
    return ''.join(clean_letter(letter) for letter in text)


def clean_letter(letter):
    alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'
    space = ' '
    lower_letter = letter.lower()
    if lower_letter in alphabet:
        return lower_letter
    return space
