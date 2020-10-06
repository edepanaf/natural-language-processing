# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import numpy as np


class Memoization:

    def __init__(self, type_hash_eq_triplets):
        self.type_hash_eq_triplets = type_hash_eq_triplets
        self.pseudo_hash_to_argument = dict()
        self.pseudo_hash_to_evaluation = dict()

    def pseudo_hash_and_eq_from_argument(self, argument):
        for my_type, my_hash, my_eq in self.type_hash_eq_triplets:
            if isinstance(argument, my_type):
                return my_hash(argument), lambda other_argument: my_eq(argument, other_argument)
        return argument.__hash__(), argument.__eq__

    def __call__(self, function):

        def memoized_function(*args):
            pseudo_hashes, pseudo_eqs = zip(self.pseudo_hash_and_eq_from_argument(argument) for argument in args)


            memorized_object = self.id_tuple_to_object.get(id_tuple)
            if (memorized_object is not None) and tuple_equality(memorized_object, args):
                return self.id_tuple_to_value[id_tuple]

    def get_memorized_value(self, argument, default=None):
        memorized_object = self.id_to_object.get(id(argument))
        if memorized_object is not None:
            if equality(argument, memorized_object):
                return memorized_object
        return default


            


def equality(u0, u1):
    if isinstance(u0, np.ndarray):
        return (u0 == u1).all()
    return u0 == u1


def tuple_equality(tuple0, tuple1):
    for item0, item1 in zip(tuple0, tuple1):
        if not equality(item0, item1):
            return False
    return True


def tuple_hash(tuple):
