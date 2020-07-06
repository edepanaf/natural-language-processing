# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


from oracle_claim import OracleClaim


class MemoryNormalize:
    def __init__(self):
        self.normalized_vector = None
        self.norm = None


class MemoryCosineDistance:
    def __init__(self):
        self.distance = None
        self.argument0 = MemoryNormalize()
        self.argument1 = MemoryNormalize()


class MemoryVectorize:
    def __init__(self):
        self.iterables = None
        self.vector = None
        self.vectorization = None
        self.norm = None


class MemoryDistance:
    def __init__(self):
        self.argument0 = MemoryVectorize()
        self.argument1 = MemoryVectorize()
        self.distance = None

    def save_cosine_distance(self, memory_cosine_distance):
        self.argument0.norm = memory_cosine_distance.argument0.norm
        self.argument1.norm = memory_cosine_distance.argument1.norm


class MemoryOracleClaim(MemoryDistance, OracleClaim):
    def __init__(self, oracle_claim):
        MemoryDistance.__init__(self)
        OracleClaim.__init__(self, oracle_claim.iterables_pair, oracle_claim.distance_interval)
