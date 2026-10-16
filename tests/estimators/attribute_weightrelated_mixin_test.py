import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris, make_friedman1
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from weighted_tree_forest.compat.sklearn_compat import check_estimator_compat
from weighted_tree_forest.estimators.attribute_weightrelated_mixin import AttributeWeightRelatedForestMixin
from weighted_tree_forest.estimators.tree_weight_calculator import (
    DecisionPathArithmeticMeanTreeWeightCalculator,
    DecisionPathTreeWeightCalculator,
    UniformTreeWeightCalculator,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from tests.estimators.estimator_factory import with_tree_weight_calculator

class AwRFCclass(AttributeWeightRelatedForestMixin, RandomForestClassifier):
    pass

class AWRfRegressor(AttributeWeightRelatedForestMixin, RandomForestRegressor):
    pass

AwRFCclassWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedForestMixin, RandomForestClassifier, "AwRFCclassWithCalculator", __name__
)
AWRfRegressorWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedForestMixin, RandomForestRegressor, "AWRfRegressorWithCalculator", __name__
)

# Bootstrapping makes sample_weight non-equivalent to repeated samples; sklearn
# xfails these checks for its own random forests, keyed by exact class only.
EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": (
        "sample_weight is not equivalent to removing/repeating samples."
    ),
    "check_sample_weight_equivalence_on_sparse_data": (
        "sample_weight is not equivalent to removing/repeating samples."
    ),
}

DECISION_PATH_CALCULATORS = {
    "geometric_mean": DecisionPathTreeWeightCalculator,
    "arithmetic_mean": DecisionPathArithmeticMeanTreeWeightCalculator,
}

class AttributeWeightRelatedForestMixinTest(unittest.TestCase):
    
    def get_estimators(self):
        return {
            "RFClassifier": AwRFCclass(n_estimators=10, random_state=42),
            "RFRegressor": AWRfRegressor(n_estimators=10, random_state=42),
            "RFClassifierWithCalculator": AwRFCclassWithCalculator(
                n_estimators=10,
                random_state=42,
                tree_weight_calculator=UniformTreeWeightCalculator(),
            ),
            "RFRegressorWithCalculator": AWRfRegressorWithCalculator(
                n_estimators=10,
                random_state=42,
                tree_weight_calculator=UniformTreeWeightCalculator(),
            ),
            **{
                f"{estimator_name}With_{calculator_name}": estimator_class(
                    n_estimators=10,
                    random_state=42,
                    tree_weight_calculator=calculator_class(),
                )
                for estimator_name, estimator_class in (
                    ("RFClassifier", AwRFCclassWithCalculator),
                    ("RFRegressor", AWRfRegressorWithCalculator),
                )
                for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items()
            },
        }

    def test_sklearn(self):

        for name, clf in self.get_estimators().items():
            with self.subTest(estimator=name):
                check_estimator_compat(clf, expected_failed_checks=EXPECTED_FAILED_CHECKS)


N_ESTIMATORS = 10

def get_datasets():
    X_clf, y_clf = load_iris(return_X_y=True)
    y_clf = np.array(["setosa", "versicolor", "virginica"])[y_clf]
    X_reg, y_reg = load_diabetes(return_X_y=True)
    return {
        "classification": (X_clf, y_clf, AwRFCclassWithCalculator),
        "multioutput_classification": (
            X_clf, np.c_[y_clf, y_clf == "setosa"], AwRFCclassWithCalculator
        ),
        "regression": (X_reg, y_reg, AWRfRegressorWithCalculator),
        "multioutput_regression": (
            X_reg, np.c_[y_reg, -y_reg], AWRfRegressorWithCalculator
        ),
    }

def as_list(proba):
    """Wraps single-output probabilities so that they match multi-output ones."""
    return proba if isinstance(proba, list) else [proba]

class WeightedPredictionTest(unittest.TestCase):

    def fit(self, X, y, estimator_class, calculator):
        return estimator_class(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            tree_weight_calculator=calculator,
        ).fit(X, y)

    def assert_matches_unweighted(self, estimator, X, attribute_weights):
        """Weighted outputs equal the unweighted forest's outputs."""
        predictions = estimator.predict(X, weights=attribute_weights)
        if hasattr(estimator, "predict_proba"):
            probas = as_list(estimator.predict_proba(X, weights=attribute_weights))
            expected_probas = as_list(estimator.predict_proba(X))
            self.assertEqual(len(probas), len(expected_probas))
            for proba, expected in zip(probas, expected_probas):
                np.testing.assert_allclose(proba, expected)
            np.testing.assert_array_equal(predictions, estimator.predict(X))
        else:
            np.testing.assert_allclose(predictions, estimator.predict(X))

    def test_decision_path_all_weights_one(self):
        # Any mean of ones is 1, so every tree gets weight 1 and the weighted
        # mean is the forest average.
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                with self.subTest(calculator=calculator_name, dataset=name):
                    estimator = self.fit(X, y, estimator_class, calculator_class())
                    self.assert_matches_unweighted(estimator, X, np.ones(X.shape[1]))

    def test_decision_path_all_weights_zero(self):
        # Every path tests at least one attribute and any mean of zeros is 0, so
        # every tree gets weight 0: probabilities fall back to uniform and
        # regression to the unweighted forest average.
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                with self.subTest(calculator=calculator_name, dataset=name):
                    estimator = self.fit(X, y, estimator_class, calculator_class())
                    attribute_weights = np.zeros(X.shape[1])
                    predictions = estimator.predict(X, weights=attribute_weights)
                    if not hasattr(estimator, "predict_proba"):
                        np.testing.assert_allclose(predictions, estimator.predict(X))
                        continue
                    probas = as_list(estimator.predict_proba(X, weights=attribute_weights))
                    for proba, classes in zip(probas, as_list(estimator.classes_)):
                        np.testing.assert_allclose(proba, 1 / len(classes))
                    # Argmax over uniform probabilities picks the first class.
                    first_classes = [classes[0] for classes in as_list(estimator.classes_)]
                    np.testing.assert_array_equal(
                        predictions.reshape(len(X), -1),
                        np.tile(first_classes, (len(X), 1)),
                    )

    def test_uniform_ignores_attribute_weights(self):
        for name, (X, y, estimator_class) in get_datasets().items():
            for weight in (0.0, 1.0):
                with self.subTest(dataset=name, weight=weight):
                    estimator = self.fit(X, y, estimator_class, UniformTreeWeightCalculator())
                    attribute_weights = np.full(X.shape[1], weight)
                    self.assert_matches_unweighted(estimator, X, attribute_weights)

    def random_attribute_weights(self, rng, n_features):
        """Yields (case, attribute_weights) pairs of random weights.

        Strictly positive weights give every tree a different, non-zero weight.
        Deep trees test almost every attribute, so even one zero attribute
        zeroes most tree weights; those cases are kept few and separate.
        """
        for repetition in range(5):
            yield f"positive_{repetition}", rng.uniform(0.1, 1.0, n_features)
        for repetition in range(3):
            attribute_weights = rng.uniform(0.1, 1.0, n_features)
            attribute_weights[rng.integers(n_features)] = 0.0
            yield f"one_zero_{repetition}", attribute_weights

    def weighted_tree_mean(self, tree_outputs, tree_weights, fallback):
        """Reference weighted mean; samples without weighted trees get `fallback`."""
        total = tree_weights.sum(axis=1)
        expected = np.empty_like(np.asarray(tree_outputs[0], dtype=float))
        for i in range(len(total)):
            if total[i] > 0:
                expected[i] = sum(w * out[i] for w, out in zip(tree_weights[i], tree_outputs)) / total[i]
            else:
                expected[i] = fallback[i] if np.ndim(fallback) else fallback
        return expected

    def assert_weighted_mean_of_trees(self, estimator, X, attribute_weights):
        """Weighted outputs equal the tree outputs averaged with the calculator's weights."""
        tree_weights = estimator.tree_weight_calculator(estimator, X, attribute_weights)
        X32 = X.astype(np.float32)
        predictions = estimator.predict(X, weights=attribute_weights)

        if hasattr(estimator, "predict_proba"):
            expected_probas = [
                self.weighted_tree_mean(
                    [as_list(tree.predict_proba(X32))[k] for tree in estimator.estimators_],
                    tree_weights,
                    1 / len(classes),
                )
                for k, classes in enumerate(as_list(estimator.classes_))
            ]
            probas = as_list(estimator.predict_proba(X, weights=attribute_weights))
            self.assertEqual(len(probas), len(expected_probas))
            for proba, expected in zip(probas, expected_probas):
                np.testing.assert_allclose(proba, expected)

            expected_predictions = np.column_stack([
                classes[np.argmax(expected, axis=1)]
                for classes, expected in zip(as_list(estimator.classes_), expected_probas)
            ])
            np.testing.assert_array_equal(
                predictions.reshape(len(X), -1), expected_predictions
            )
        else:
            tree_predictions = [tree.predict(X32) for tree in estimator.estimators_]
            expected = self.weighted_tree_mean(
                tree_predictions, tree_weights, estimator.predict(X)
            )
            np.testing.assert_allclose(predictions, expected)

    def test_regression_predictions_are_informative(self):
        # Held-out R^2 of predictions unrelated to the target is at most about
        # 0; weighted predictions must stay well above it.
        X, y = make_friedman1(n_samples=600, random_state=0)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        rng = np.random.default_rng(42)
        calculators = {"uniform": UniformTreeWeightCalculator, **DECISION_PATH_CALCULATORS}
        for calculator_name, calculator_class in calculators.items():
            estimator = AWRfRegressorWithCalculator(
                n_estimators=50, random_state=0, tree_weight_calculator=calculator_class()
            ).fit(X_train, y_train)
            cases = {
                "ones": np.ones(X.shape[1]),
                "zeros": np.zeros(X.shape[1]),
                **{f"random_{i}": rng.uniform(0.1, 1.0, X.shape[1]) for i in range(3)},
            }
            for case, attribute_weights in cases.items():
                with self.subTest(calculator=calculator_name, case=case):
                    r2 = r2_score(y_test, estimator.predict(X_test, weights=attribute_weights))
                    self.assertGreater(r2, 0.3)

    def test_predict_proba_is_probability_distribution(self):
        rng = np.random.default_rng(42)
        calculators = {"uniform": UniformTreeWeightCalculator, **DECISION_PATH_CALCULATORS}
        for calculator_name, calculator_class in calculators.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                estimator = self.fit(X, y, estimator_class, calculator_class())
                if not hasattr(estimator, "predict_proba"):
                    continue
                cases = {
                    "ones": np.ones(X.shape[1]),
                    "zeros": np.zeros(X.shape[1]),
                    **dict(self.random_attribute_weights(rng, X.shape[1])),
                }
                for case, attribute_weights in cases.items():
                    with self.subTest(calculator=calculator_name, dataset=name, case=case):
                        probas = as_list(estimator.predict_proba(X, weights=attribute_weights))
                        for proba, classes in zip(probas, as_list(estimator.classes_)):
                            self.assertEqual(proba.shape, (len(X), len(classes)))
                            self.assertTrue(np.all((proba >= 0) & (proba <= 1)))
                            np.testing.assert_allclose(proba.sum(axis=1), 1.0)

    def test_decision_path_random_weights(self):
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            rng = np.random.default_rng(42)
            for name, (X, y, estimator_class) in get_datasets().items():
                estimator = self.fit(X, y, estimator_class, calculator_class())
                for case, attribute_weights in self.random_attribute_weights(rng, X.shape[1]):
                    with self.subTest(calculator=calculator_name, dataset=name, case=case):
                        self.assert_weighted_mean_of_trees(estimator, X, attribute_weights)

    def test_uniform_random_weights(self):
        rng = np.random.default_rng(42)
        for name, (X, y, estimator_class) in get_datasets().items():
            estimator = self.fit(X, y, estimator_class, UniformTreeWeightCalculator())
            for case, attribute_weights in self.random_attribute_weights(rng, X.shape[1]):
                with self.subTest(dataset=name, case=case):
                    self.assert_matches_unweighted(estimator, X, attribute_weights)
