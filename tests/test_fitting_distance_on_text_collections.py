import unittest


from fitting_distance_on_text_collections import *
from oracle_claim import OracleClaim


class TestFittingDistanceOnTextCollections(unittest.TestCase):

    def test_fit_of_fitting_distance_on_text_collections(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingDistanceOnTextCollections(text_collection)
        oracle_claim = OracleClaim(({text1}, {text2}), (0.1, 0.2))
        old_distance12 = distance({text1}, {text2})
        distance.fit({oracle_claim})
        new_distance12 = distance({text1}, {text2})
        self.assertTrue(old_distance12 > new_distance12)

    def test_distance_on_text_collections(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingDistanceOnTextCollections(text_collection)
        self.assertTrue(isinstance(distance({text0, text1}, {text1, text2}), float))

    def test_type_bag_of_factors_from_text(self):
        bag = bag_of_factors_from_text('wefwef')
        self.assertTrue(isinstance(bag, tuple))

    def test_bag_of_factors_from_text(self):
        text = 'ab c  ab '
        bag = bag_of_factors_from_text(text, max_factor_length=3)
        expected = {'a': 2, 'ab': 2, 'ab ': 2, 'b': 2, 'b ': 2, 'b c': 1, ' ': 4, ' c': 1,
                    ' c ': 1, 'c': 1, 'c ': 1, 'c  ': 1, '  ': 1, '  a': 1, ' a': 1, ' ab': 1}
        computed = dict()
        for factor in bag:
            computed[factor] = computed.get(factor, 0) + 1
        self.assertEqual(computed, expected)

    def test_clean_text(self):
        text = 'A;b_C'
        clean = 'a b c'
        self.assertEqual(clean_text(text), clean)

    def test_clean_letter(self):
        self.assertEqual(clean_letter('a'), 'a')
        self.assertEqual(clean_letter(','), ' ')

    def test_non_existent_weight_from_factor(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        distance = FittingDistanceOnTextCollections(text_collection)
        self.assertEqual(distance.weight_from_factor('wef'), 0.)

    def test_input_bag_dictionary(self):
        bag_to_weight = {'aa': 1., 'ab': 2., 'abc': 3.}
        distance = FittingDistanceOnTextCollections(bag_to_weight)
        self.assertEqual(distance.weight_from_text('aa'), 1.)
        self.assertEqual(distance.weight_from_text('ab'), 2.)
        self.assertEqual(distance.weight_from_text('abc'), 3.)

    def test_input_factor_dictionary(self):
        text0 = 'banana'
        text1 = 'ananas'
        text2 = 'bans'
        text_collection = {text0, text1, text2}
        factor_to_weight = {'a': 1., 'b': 2., 'ba': 3., 'c': 28.}
        distance = FittingDistanceOnTextCollections(text_collection, factor_to_weight=factor_to_weight)
        self.assertEqual(distance.weight_from_factor('a'), 1.)
        self.assertEqual(distance.weight_from_factor('ba'), 3.)
        self.assertEqual(distance.weight_from_factor('d'), 0.)
        self.assertEqual(distance.weight_from_factor('c'), 0.)


if __name__ == '__main__':
    unittest.main()
