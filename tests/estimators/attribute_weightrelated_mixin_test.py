import inspect
import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from weighted_tree_forest.compat.sklearn_compat import check_estimator_compat
from weighted_tree_forest.estimators.attribute_weightrelated_mixin import AttributeWeightRelatedForestMixin
from weighted_tree_forest.estimators.tree_weight_calculator import (
    DecisionPathArithmeticMeanTreeWeightCalculator,
    DecisionPathTreeWeightCalculator,
    UniformTreeWeightCalculator,
)

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class AwRFCclass(AttributeWeightRelatedForestMixin, RandomForestClassifier):
    pass

class AWRfRegressor(AttributeWeightRelatedForestMixin, RandomForestRegressor):
    pass

def with_tree_weight_calculator(forest_class, name):
    """Subclass of the mixin and `forest_class` taking a tree_weight_calculator.

    The constructor signature is copied from the installed scikit-learn, so it
    follows the forest parameters of each version (e.g. monotonic_cst since 1.4).
    """
    parent_init = forest_class.__init__
    signature = inspect.signature(parent_init)

    def __init__(self, *args, tree_weight_calculator=None, **kwargs):
        parent_init(self, *args, **kwargs)
        self.tree_weight_calculator = tree_weight_calculator

    __init__.__signature__ = signature.replace(
        parameters=[
            *signature.parameters.values(),
            inspect.Parameter(
                "tree_weight_calculator", inspect.Parameter.KEYWORD_ONLY, default=None
            ),
        ]
    )
    # Module-level name and qualname keep instances picklable.
    return type(
        name,
        (AttributeWeightRelatedForestMixin, forest_class),
        {"__init__": __init__, "__module__": __name__, "__qualname__": name},
    )

AwRFCclassWithCalculator = with_tree_weight_calculator(
    RandomForestClassifier, "AwRFCclassWithCalculator"
)
AWRfRegressorWithCalculator = with_tree_weight_calculator(
    RandomForestRegressor, "AWRfRegressorWithCalculator"
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

    def assert_predictions_scaled(self, estimator, X, attribute_weights, scale):
        """Weighted outputs equal `scale` times the unweighted forest's average."""
        predictions = estimator.predict(X, weights=attribute_weights)
        if hasattr(estimator, "predict_proba"):
            probas = as_list(estimator.predict_proba(X, weights=attribute_weights))
            expected_probas = as_list(estimator.predict_proba(X))
            self.assertEqual(len(probas), len(expected_probas))
            for proba, expected in zip(probas, expected_probas):
                np.testing.assert_allclose(proba, scale * expected)
            if scale > 0:
                np.testing.assert_array_equal(predictions, estimator.predict(X))
        else:
            np.testing.assert_allclose(predictions, scale * estimator.predict(X))

    def test_decision_path_all_weights_one(self):
        # Any mean of ones is 1, so every tree gets weight 1 and the weighted
        # sum is N_ESTIMATORS times the forest average.
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                with self.subTest(calculator=calculator_name, dataset=name):
                    estimator = self.fit(X, y, estimator_class, calculator_class())
                    self.assert_predictions_scaled(estimator, X, np.ones(X.shape[1]), N_ESTIMATORS)

    def test_decision_path_all_weights_zero(self):
        # Every path tests at least one attribute and any mean of zeros is 0, so
        # every tree gets weight 0.
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                with self.subTest(calculator=calculator_name, dataset=name):
                    estimator = self.fit(X, y, estimator_class, calculator_class())
                    attribute_weights = np.zeros(X.shape[1])
                    self.assert_predictions_scaled(estimator, X, attribute_weights, 0)
                    if hasattr(estimator, "predict_proba"):
                        # Argmax over all-zero sums picks the first class.
                        predictions = estimator.predict(X, weights=attribute_weights)
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
                    self.assert_predictions_scaled(estimator, X, attribute_weights, N_ESTIMATORS)

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

    def assert_weighted_sum_of_trees(self, estimator, X, attribute_weights):
        """Weighted outputs equal the tree outputs summed with the calculator's weights."""
        tree_weights = estimator.tree_weight_calculator(estimator, X, attribute_weights)
        X32 = X.astype(np.float32)
        predictions = estimator.predict(X, weights=attribute_weights)

        if hasattr(estimator, "predict_proba"):
            expected_probas = [
                sum(
                    tree_weights[:, [t]] * as_list(tree.predict_proba(X32))[k]
                    for t, tree in enumerate(estimator.estimators_)
                )
                for k in range(estimator.n_outputs_)
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
            expected = sum(
                tree_weights[:, t].reshape((-1,) + (1,) * (predictions.ndim - 1))
                * tree.predict(X32)
                for t, tree in enumerate(estimator.estimators_)
            )
            np.testing.assert_allclose(predictions, expected)

    def test_decision_path_random_weights(self):
        for calculator_name, calculator_class in DECISION_PATH_CALCULATORS.items():
            rng = np.random.default_rng(42)
            for name, (X, y, estimator_class) in get_datasets().items():
                estimator = self.fit(X, y, estimator_class, calculator_class())
                for case, attribute_weights in self.random_attribute_weights(rng, X.shape[1]):
                    with self.subTest(calculator=calculator_name, dataset=name, case=case):
                        self.assert_weighted_sum_of_trees(estimator, X, attribute_weights)

    def test_uniform_random_weights(self):
        rng = np.random.default_rng(42)
        for name, (X, y, estimator_class) in get_datasets().items():
            estimator = self.fit(X, y, estimator_class, UniformTreeWeightCalculator())
            for case, attribute_weights in self.random_attribute_weights(rng, X.shape[1]):
                with self.subTest(dataset=name, case=case):
                    self.assert_predictions_scaled(estimator, X, attribute_weights, N_ESTIMATORS)
