# natural-language-processing
# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>

"""
We use mathsage to test the following identity.

Let $s_0$, $s_1$ denote the indicator vectors of two sets of bags,
$b$ the weight vector of bags, $f$ the weight vector of items,
$M$ the item-weight occurrence matrix.
Let $v$ denote the vectorization $f \odot M \cdot (b \odot s)$ of $s$
and $\alpha$ the cosine distance between the vectors $v_0$ and $v_1$.
When $\epsilon$ and $\delta$ are two small vectors,
we consider the new weights for items and bags
$f + f \odot \epsilon$ and $b + b \odot \delta$.
The cosine distance between the new vectorizations of $s_0$ and $s_1$
is denoted by $\beta$.

The identity we check using mathsage is
\[
    \alpha - \beta =
    x^T \epsilon
    + y^T \delta
    + \bigO(\|\epsilon\| + \|\delta\|)^2
\]
where
\begin{align*}
    w &= \frac{v}{\|v\|},
    \\
    v_{0,1} &= w_0 - (1 - \alpha) w_1,
    \\
    v_{1,0} &= w_1 - (1 - \alpha) w_0,
    \\
    x &= v_{0,1} \odot w_1 + v_{1,0} \odot w_0
    \\
    y &=
        \frac{1}{\|v_1\|} s_1 \odot b \odot \transpose{M} (f \odot v_{0,1})
        + \frac{1}{\|v_0\|} s_0 \odot b \odot \transpose{M} (f \odot v_{1,0})
\end{align*}



def random_matrix(rows, columns):
    return matrix([[random() for _ in range(columns)] for __ in range(rows)])


def random_vector(dimension):
    return vector([random() for _ in range(dimension)])


def hadamard_product(v0, v1):
    if len(v0) != len(v1):
        raise ValueError
    return vector([v0[i] * v1[i] for i in range(len(v0))])


def cosine_distance(v0, v1):
    return 1 - v0 * v1 / v0.norm() / v1.norm()


def vectorization(witems, M, wbags, s):
    return hadamard_product(witems, M * hadamard_product(wbags, s))


def isclose(x, y, precision=10**(-8)):
    return bool(abs(x-y) < precision)


# diagonal_matrix(v)
# M.tranpose()
# v.norm()


nbags = 3
nitems = 5


M = random_matrix(nitems, nbags)
wbags = random_vector(nbags)
witems = random_vector(nitems)
s0 = random_vector(nbags)
s1 = random_vector(nbags)
var('xbags', 'xitems')
epsilon = xitems * random_vector(nitems)
delta = xbags * random_vector(nbags)


v0 = vectorization(witems, M, wbags, s0)
v1 = vectorization(witems, M, wbags, s1)
nwbags = wbags + hadamard_product(wbags, delta)
nwitems = witems + hadamard_product(witems, epsilon)
nv0 = vectorization(nwitems, M, nwbags, s0)
nv1 = vectorization(nwitems, M, nwbags, s1)


current_distance = cosine_distance(v0, v1)
target_distance = cosine_distance(nv0, nv1)


pitems = taylor(target_distance.subs(xbags == 0), xitems, 0, 1)
pbags = taylor(target_distance.subs(xitems == 0), xbags, 0, 1)


print(isclose(pitems.coefficient(xitems, 0), current_distance))
print(isclose(pbags.coefficient(xbags, 0), current_distance))


error_term_items = pitems.coefficient(xitems, 1)
error_term_bags = pbags.coefficient(xbags, 1)


vv0 = v0 / v0.norm()
vv1 = v1 / v1.norm()
vv01 = vv0 - (1 - current_distance)*vv1
vv10 = vv1 - (1 - current_distance)*vv0
formula_error_term_items = - epsilon * (hadamard_product(vv01, vv1) + hadamard_product(vv10, vv0))
formula_error_term_bags = - delta * (1/v1.norm() * hadamard_product(hadamard_product(s1, wbags), M.transpose() * hadamard_product(witems, vv01)) + 1/v0.norm() * hadamard_product(hadamard_product(s0, wbags), M.transpose() * hadamard_product(witems, vv10)))
formula_error_term_items = formula_error_term_items.coefficient(xitems, 1)
formula_error_term_bags = formula_error_term_bags.coefficient(xbags, 1)


print(isclose(error_term_items, formula_error_term_items))
print(isclose(error_term_bags, formula_error_term_bags))

"""
