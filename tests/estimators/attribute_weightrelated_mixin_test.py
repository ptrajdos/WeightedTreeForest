import unittest
import numpy as np
from sklearn.datasets import load_diabetes, load_iris
from sklearn.utils.estimator_checks import check_estimator
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

class AwRFCclassWithCalculator(AttributeWeightRelatedForestMixin, RandomForestClassifier):
    def __init__(
        self,
        n_estimators=100,
        *,
        tree_weight_calculator=None,
        criterion="gini",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features="sqrt",
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        class_weight=None,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        self.tree_weight_calculator = tree_weight_calculator

class AWRfRegressorWithCalculator(AttributeWeightRelatedForestMixin, RandomForestRegressor):
    def __init__(
        self,
        n_estimators=100,
        *,
        tree_weight_calculator=None,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=1.0,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None,
        monotonic_cst=None,
    ):
        super().__init__(
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            ccp_alpha=ccp_alpha,
            max_samples=max_samples,
            monotonic_cst=monotonic_cst,
        )
        self.tree_weight_calculator = tree_weight_calculator

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
                check_estimator(clf, expected_failed_checks=EXPECTED_FAILED_CHECKS)


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
