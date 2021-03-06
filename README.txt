# natural-language-processing
# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>



============ Summary ============

This Python package provides a class 'FittingDistance' which implements a distance on sets of bags of items
(i.e. collections of collections of items), able to evolve to fit examples.
Its main application is expected to be on Natural Language Processing.
In that case, the bags are texts and items might be factors (another popular choice is lemmatized words).
We provide a specialized class 'FittingDistanceOnTextCollections' for this case.


============ Fitting distance on text collections ============


from 



============ Summary ============


This Python package provides a distance on collections of bags of items
that is able to evolve when true distance instances are provided.
Its main application is expected to be on Natural Language Processing.
It unifies the 'bag of words' and 'bag of factors' approaches.
Indeed, a text can be transformed into a collection of words after lemmatization,
or into a the collection of its factors (bounding the length of those factors
is then recommended).

The LearningDistance object is initialized by providing
the collection of all bags considered.
Those bags must be hashable iterables.
The distance itself is computed using weights on the items and bags
(a large weight corresponding to an important item or bag).
Those weights can be provided by the user.
Otherwise, they are computed following the tf-idf heuristic.

The LearningDistance object is also able to learn.
Let us define an 'oracle claim' as a pair of collections of bags
and an interval for the true distance between them.
The LearningDistance object can use a collection of oracle claims
to change the item and bag weights so that
the distance conforms better with the oracle claims.

The distance between two collections of bags is defined as
the cosine distance between their vectorizations.
The vectorization of a collection of bags is a vector on the space
of all items, which coefficients depends on the item and bag weights,
following the classic 'bag of words' and 'tf-idf' approaches.

All the computations rely on sparse matrix manipulations, implemented by scipy.



============ Tutorial ============


# Import the main class.

from learning_distance import LearningDistance

# Define the bags we are working on. They must be hashable.

bag0 = ('a', 'b', 'a', 'a')
bag1 = ('a', 'a', 'a', 'b')
bag2 = ('a', 'b', 'b', 'a')
bag3 = ('a', 'c')
all_bags = [bag0, bag1, bag2, bag3]

# The order of the items does not impact the distance computation,
# but their multiplicity does.

# Define weights on the items and bags.

item_weights = {'a': 1., 'b': 2., 'c': 0.5}
bag_weights = {bag0: 4.5, bag1: 1., bag2: 3., bag3: 2.5}

# Define the LearningDistance object.

learning_distance = LearningDistance(all_bags, weight_from_item=item_weights, weight_from_bag=bag_weights)

# If the 'item_weights' and / or 'bag_weights' are omitted,
# default weights will be computed using the tf-idf heuristic.
# We can access the weights in the following way.

learning_distance.get_item_weights()
# {'a': 0.2857142857142857, 'b': 0.5714285714285714, 'c': 0.14285714285714285}

# The weights are different from the ones we entered!
# Indeed, they are normalized to sum to '1.',
# since the distance is invariant by scalar multiplication of the weights.

# The LearningDistance object is callable and returns
# the distance between its arguments.
# Recall that those arguments are collections of bags,
# and not bags alone.

learning_distance({bag0, bag1}, {bag1, bag3})
# 0.04990974137709514

# Maybe we find this distance too low and expected a value
# between 0.1 and 0.2 instead. To express it, we make the following oracle claim.

from oracle_claim import OracleClaim
oracle_claim = OracleClaim(({bag0, bag1}, {bag1, bag3}), (0.1, 0.2))

# We can now let learning_distance learn from this claim
# and adjust its weights on items and bags.
# Let us say that we want the changes to be mainly on the item weights,
# then we write

set_of_claims = {oracle_claim}
learning_distance.learn(set_of_claims, ratio_item_bag_learning=1.)

# The collection of claims can be a set or a list.
# We can now check that the distance has changed.

learning_distance({bag0, bag1}, {bag1, bag3})
# 0.0989203613583286

# The item and bag weights can be accessed either as dictionary

learning_distance.get_item_weights()
# {'a': 0.1958114148377036, 'b': 0.6281455477367703, 'c': 0.1800337915490111}

learning_distance.get_bag_weights()
"""{('a', 'b', 'a', 'a'): 0.4090909090909091,
 ('a', 'a', 'a', 'b'): 0.09090909090909091,
 ('a', 'b', 'b', 'a'): 0.2727272727272727,
 ('a', 'c'): 0.22727272727272727}"""

# or as vectors, using directly the attributes

learning_distance.item_weights_vector
# array([0.19581141, 0.62814555, 0.18003379])

learning_distance.bag_weights_vector
# array([0.40909091, 0.09090909, 0.27272727, 0.22727273])

# This second option is faster, but we do not know to which item (or bag)
# correspond the indices of the vectors.
# Those indices can be obtained using the dictionaries

learning_distance.item_to_index
# {'a': 0, 'b': 1, 'c': 2}

learning_distance.bag_to_index
"""{('a', 'b', 'a', 'a'): 0,
    ('a', 'a', 'a', 'b'): 1,
    ('a', 'b', 'b', 'a'): 2,
    ('a', 'c'): 3}"""



============ Structure ============


--- matrix_operations.py ---

Define the functions manipulating vectors and matrices.
Rely on scipy sparse matrices.
Only this file needs changing if another implementation is chosen in the future.


--- vector_space.py ---

Define the class 'VectorSpace', initialized using a collection of bags,
and transforming item or bag collections into vectors.
Provide the methods
    __init__(bags)
    item_vector_from_dict(self, item_distribution)
    bag_vector_from_dict(self, bag_distribution)
    item_dict_from_vector(self, item_vector)
    bag_dict_from_vector(self, bag_vector)
    bag_vector_from_collection(self, bags)
    count_bags_containing_item(self, item)


--- distance.py ---

Define the class 'Distance'. Objects of this class are callable.
They input pairs of collections of bags and output their distance.
Provide the methods
    __init__(self, bags, weight_from_item=None, weight_from_bag=None)
    def __call__(self, bags0, bags1)
    __call__(self, bags)
    set_item_weights(self, weight_from_item)
    set_bag_weights(self, weight_from_bag)
    get_item_weights(self)
    get_bag_weights(self)
    tfidf_item_weights(self)
    verbose_distance(self, bags0, bags1)
    verbose_vectorize(self, bags)


--- oracle_claim.py ---

Define the class 'OracleClaim', which is used to provide
bounds on the distance between two collections of bags.
Provide the method
    __init__(self, pair_of_bag_collections, distance_interval)


--- learning_distance.py ---

Define the class 'LearningDistance', which inherits from 'Distance'.
Add the functionality to learn from 'OracleClaim' objects.
Provide the methods
    __init__(self, bags, weight_from_item=None, weight_from_bag=None)
    learn(self, oracle_claims, ratio_item_bag_learning=0.5, convergence_speed=0.5,
          number_of_iterations=DEFAULT_NUMBER_OF_ITERATIONS)
    learning_loop_on_oracle_claims(self, oracle_claims, ratio_item_bag_learning=0.5, effort=1.)
    apply_gradient_descent_step_for_one_oracle_claim(self, oracle_claim, ratio_item_bag_learning=0.5, effort=1.)
    compute_rescaling_vectors(self, enriched_oracle_claim, ratio_item_bag_learning)

Also define the class 'EnrichedOracleClaim', used to avoid
duplicate computations during the treatment of an oracle claim.


--- information_theory.py ---

The Jensen-Shannon divergence between two distributions X, Y on the same set is defined as follows.
Consider a random Boolean B and a variable Z equal to X if B is zero, Y otherwise.
The Jensen-Shannon divergence is then defined as the mutual information between B and Z.
Its square root is a metric distance.

This file defines the function 'get_jensen_shannon_distance_and_variance'
that inputs two lists of equal length of nonnegative floats summing to 1.
and returns their jensen-shannon distance as well as the associated variance
(variance of the corresponding mutual information).


--- memory.py ---

Define various classes to store intermediate computations
as side effect of calling some of the costly functions.


--- tests ---

Contain the unittests for the various files.
