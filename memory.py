# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


class MemoryNorm:
    def __init__(self):
        self.norm = None


class MemoryArgumentsNorm:
    def __init__(self):
        self.argument0 = MemoryNorm()
        self.argument1 = MemoryNorm()


class MemoryVector:
    def __init__(self):
        self.vector = None


class MemoryVectorVectorizationNorm(MemoryVector, MemoryNorm):
    def __init__(self):
        MemoryVector.__init__(self)
        MemoryNorm.__init__(self)
        self.vectorization = None


class MemoryArgumentsVectorsVectorizationsNorms (MemoryArgumentsNorm):
    def __init__(self):
        super().__init__()
        self.argument0 = MemoryVectorVectorizationNorm()
        self.argument1 = MemoryVectorVectorizationNorm()
