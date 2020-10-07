# © 2020 Nokia
# Licensed under the BSD 3-Clause License.
# SPDX-License-Identifier: BSD-3-Clause
# !/usr/bin/env python3
# coding: utf-8
# Author: Élie de Panafieu  <elie.de_panafieu@nokia-bell-labs.com>


import json
import wikipedia
from fitting_distance_on_text_collections import FittingDistanceOnTextCollections
import time


FILE_WIKIPEDIA_ARTICLES = 'wikipedia_articles.json'


def random_wikipedia_article():
    title = wikipedia.random(pages=1)
    try:
        return wikipedia.page(title).content
    except wikipedia.exceptions.DisambiguationError:
        # wikipedia.exceptions.DisambiguationError as e:
        # sometimes, the line wikipedia.page(e.options[0]).content raises again a DisambiguationError exception,
        # which is unfortunate (see for example the wikipedia page 'Shadi')
        return random_wikipedia_article()


def save_wikipedia_articles(number_of_articles, file=FILE_WIKIPEDIA_ARTICLES):
    articles = [random_wikipedia_article() for _ in range(number_of_articles)]
    open(file, 'w').write(json.dumps(articles))


def load_wikipedia_articles(file=FILE_WIKIPEDIA_ARTICLES):
    return json.loads(open(file, 'r').read())


def benchmark_fitting_distance(number_of_articles, new_articles=False):
    starting_time = time.time()
    if new_articles:
        save_wikipedia_articles(number_of_articles)
    articles = load_wikipedia_articles()
    load_time = time.time()
    print('Articles downloaded, time ' + str(load_time - starting_time))
    distance = FittingDistanceOnTextCollections(articles)
    construction_time = time.time()
    print('Distance built, time ' + str(construction_time - load_time))
    all_distances = dict()
    for i in range(len(articles)):
        for j in range(i+1, len(articles)):
            all_distances[i, j] = distance({articles[i]}, {articles[j]})
            # print(str(i) + ', ' + str(j))
    computations_time = time.time()
    print('All distances computed, time ' + str(computations_time - construction_time))
    return all_distances
