import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from weighted_tree_forest.estimators.tree_weight_calculator import DecisionPathTreeWeightCalculator

from tests.estimators.attribute_weightrelated_mixin_test import (
    AwRFCclassWithCalculator,
    AWRfRegressorWithCalculator,
)

N_ESTIMATORS = 10

def get_datasets():
    X_clf, y_clf = load_iris(return_X_y=True)
    X_reg, y_reg = load_diabetes(return_X_y=True)
    return {
        "classification": (X_clf, y_clf, RandomForestClassifier),
        "regression": (X_reg, y_reg, RandomForestRegressor),
    }

def reference_tree_weights(forest, X, attribute_weights):
    """Geometric mean of the tested attribute weights, one path at a time."""
    tree_weights = np.ones((len(X), len(forest.estimators_)))
    for t, tree in enumerate(forest.estimators_):
        visited = tree.decision_path(X.astype(np.float32)).toarray().astype(bool)
        for i in range(len(X)):
            features = tree.tree_.feature[visited[i]]
            features = features[features >= 0]
            if len(features) > 0:
                tree_weights[i, t] = np.prod(attribute_weights[features]) ** (1 / len(features))
    return tree_weights

class DecisionPathTreeWeightCalculatorTest(unittest.TestCase):

    def setUp(self):
        self.calculator = DecisionPathTreeWeightCalculator()
        self.rng = np.random.default_rng(42)

    def fit(self, X, y, forest_class, **params):
        return forest_class(n_estimators=N_ESTIMATORS, random_state=42, **params).fit(X, y)

    def test_matches_reference(self):
        for name, (X, y, forest_class) in get_datasets().items():
            with self.subTest(dataset=name):
                forest = self.fit(X, y, forest_class, max_depth=5)
                attribute_weights = self.rng.uniform(0.1, 2.0, X.shape[1])
                tree_weights = self.calculator(forest, X, attribute_weights)
                self.assertEqual(tree_weights.shape, (len(X), N_ESTIMATORS))
                np.testing.assert_allclose(
                    tree_weights, reference_tree_weights(forest, X, attribute_weights)
                )

    def test_stumps_use_root_attribute_weight(self):
        # A depth-1 tree tests only its root attribute, on every path.
        for name, (X, y, forest_class) in get_datasets().items():
            with self.subTest(dataset=name):
                forest = self.fit(X, y, forest_class, max_depth=1)
                attribute_weights = self.rng.uniform(0.1, 2.0, X.shape[1])
                root_features = [tree.tree_.feature[0] for tree in forest.estimators_]
                np.testing.assert_allclose(
                    self.calculator(forest, X, attribute_weights),
                    np.tile(attribute_weights[root_features], (len(X), 1)),
                )

    def test_zero_weight_zeroes_paths_testing_attribute(self):
        for name, (X, y, forest_class) in get_datasets().items():
            with self.subTest(dataset=name):
                forest = self.fit(X, y, forest_class, max_depth=5)
                attribute_weights = self.rng.uniform(0.1, 2.0, X.shape[1])
                attribute_weights[0] = 0.0
                tree_weights = self.calculator(forest, X, attribute_weights)
                for t, tree in enumerate(forest.estimators_):
                    visited = tree.decision_path(X.astype(np.float32)).toarray().astype(bool)
                    tests_attribute = (visited & (tree.tree_.feature == 0)).any(axis=1)
                    np.testing.assert_array_equal(tree_weights[:, t] == 0, tests_attribute)

    def test_scaling_attribute_weights_scales_tree_weights(self):
        # The geometric mean is homogeneous: scaling all inputs scales the mean.
        for name, (X, y, forest_class) in get_datasets().items():
            with self.subTest(dataset=name):
                forest = self.fit(X, y, forest_class, max_depth=5)
                attribute_weights = self.rng.uniform(0.1, 2.0, X.shape[1])
                np.testing.assert_allclose(
                    self.calculator(forest, X, 3.0 * attribute_weights),
                    3.0 * self.calculator(forest, X, attribute_weights),
                )

    def test_single_leaf_tree_has_weight_one(self):
        X = self.rng.normal(size=(20, 3))
        forest = self.fit(X, np.ones(20), RandomForestRegressor)
        np.testing.assert_array_equal(
            self.calculator(forest, X, np.array([0.0, 2.0, 3.0])),
            np.ones((20, N_ESTIMATORS)),
        )

    def test_invalid_attribute_weights(self):
        X, y, forest_class = get_datasets()["classification"]
        forest = self.fit(X, y, forest_class)
        for name, attribute_weights in {
            "negative": np.array([-1.0, 1.0, 1.0, 1.0]),
            "wrong_length": np.ones(X.shape[1] + 1),
        }.items():
            with self.subTest(case=name):
                with self.assertRaises(ValueError):
                    self.calculator(forest, X, attribute_weights)

class DecisionPathWeightedPredictionTest(unittest.TestCase):

    def test_predictions_are_weighted_sums_of_tree_outputs(self):
        rng = np.random.default_rng(42)
        datasets = {
            "classification": (*load_iris(return_X_y=True), AwRFCclassWithCalculator),
            "regression": (*load_diabetes(return_X_y=True), AWRfRegressorWithCalculator),
        }
        for name, (X, y, estimator_class) in datasets.items():
            with self.subTest(dataset=name):
                estimator = estimator_class(
                    n_estimators=N_ESTIMATORS,
                    random_state=42,
                    max_depth=5,
                    tree_weight_calculator=DecisionPathTreeWeightCalculator(),
                ).fit(X, y)
                attribute_weights = rng.uniform(0.1, 2.0, X.shape[1])
                tree_weights = reference_tree_weights(estimator, X, attribute_weights)
                X32 = X.astype(np.float32)

                if hasattr(estimator, "predict_proba"):
                    expected_proba = sum(
                        tree_weights[:, [t]] * tree.predict_proba(X32)
                        for t, tree in enumerate(estimator.estimators_)
                    )
                    np.testing.assert_allclose(
                        estimator.predict_proba(X, weights=attribute_weights), expected_proba
                    )
                    np.testing.assert_array_equal(
                        estimator.predict(X, weights=attribute_weights),
                        estimator.classes_[np.argmax(expected_proba, axis=1)],
                    )
                else:
                    expected = sum(
                        tree_weights[:, t] * tree.predict(X32)
                        for t, tree in enumerate(estimator.estimators_)
                    )
                    np.testing.assert_allclose(
                        estimator.predict(X, weights=attribute_weights), expected
                    )
