import unittest
import numpy as np
from scipy.special import expit, softmax
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from weighted_tree_forest.compat.sklearn_compat import check_estimator_compat
from weighted_tree_forest.estimators.attribute_weightrelated_boosting_mixin import (
    AttributeWeightRelatedBoostingMixin,
)
from weighted_tree_forest.estimators.tree_weight_calculator import (
    DecisionPathArithmeticMeanTreeWeightCalculator,
    DecisionPathTreeWeightCalculator,
    TreeWeightCalculator,
    UniformTreeWeightCalculator,
)

from tests.estimators.estimator_factory import with_tree_weight_calculator

class AwGBClassifier(AttributeWeightRelatedBoostingMixin, GradientBoostingClassifier):
    pass

class AwGBRegressor(AttributeWeightRelatedBoostingMixin, GradientBoostingRegressor):
    pass

class AwAdaBoostClassifier(AttributeWeightRelatedBoostingMixin, AdaBoostClassifier):
    pass

class AwAdaBoostRegressor(AttributeWeightRelatedBoostingMixin, AdaBoostRegressor):
    pass

class AwHistGBRegressor(AttributeWeightRelatedBoostingMixin, HistGradientBoostingRegressor):
    pass

AwGBClassifierWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedBoostingMixin, GradientBoostingClassifier,
    "AwGBClassifierWithCalculator", __name__,
)
AwGBRegressorWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedBoostingMixin, GradientBoostingRegressor,
    "AwGBRegressorWithCalculator", __name__,
)
AwAdaBoostClassifierWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedBoostingMixin, AdaBoostClassifier,
    "AwAdaBoostClassifierWithCalculator", __name__,
)
AwAdaBoostRegressorWithCalculator = with_tree_weight_calculator(
    AttributeWeightRelatedBoostingMixin, AdaBoostRegressor,
    "AwAdaBoostRegressorWithCalculator", __name__,
)

CALCULATORS = {
    "uniform": UniformTreeWeightCalculator,
    "geometric_mean": DecisionPathTreeWeightCalculator,
    "arithmetic_mean": DecisionPathArithmeticMeanTreeWeightCalculator,
}

N_ESTIMATORS = 10

# sklearn xfails these checks for its own boosting ensembles, keyed by exact
# class only.
EXPECTED_FAILED_CHECKS = {
    "check_sample_weight_equivalence_on_dense_data": (
        "sample_weight is not equivalent to removing/repeating samples."
    ),
    "check_sample_weight_equivalence_on_sparse_data": (
        "sample_weight is not equivalent to removing/repeating samples."
    ),
}

class FixedTreeWeightCalculator(TreeWeightCalculator):
    """Returns the given tree weights, ignoring the attribute weights."""

    def __init__(self, tree_weights):
        self.tree_weights = tree_weights

    def calculate_tree_weights(self, forest, X, attribute_weights):
        return self.tree_weights

class AttributeWeightRelatedBoostingMixinTest(unittest.TestCase):

    def get_estimators(self):
        plain = {
            "GBClassifier": AwGBClassifier(n_estimators=N_ESTIMATORS, random_state=42),
            "GBRegressor": AwGBRegressor(n_estimators=N_ESTIMATORS, random_state=42),
            "AdaBoostClassifier": AwAdaBoostClassifier(n_estimators=N_ESTIMATORS, random_state=42),
            "AdaBoostRegressor": AwAdaBoostRegressor(n_estimators=N_ESTIMATORS, random_state=42),
        }
        with_calculator = {
            f"{estimator_name}With_{calculator_name}": estimator_class(
                n_estimators=N_ESTIMATORS,
                random_state=42,
                tree_weight_calculator=calculator_class(),
            )
            for estimator_name, estimator_class in (
                ("GBClassifier", AwGBClassifierWithCalculator),
                ("GBRegressor", AwGBRegressorWithCalculator),
                ("AdaBoostClassifier", AwAdaBoostClassifierWithCalculator),
                ("AdaBoostRegressor", AwAdaBoostRegressorWithCalculator),
            )
            for calculator_name, calculator_class in CALCULATORS.items()
        }
        return {**plain, **with_calculator}

    def test_sklearn(self):
        for name, estimator in self.get_estimators().items():
            with self.subTest(estimator=name):
                check_estimator_compat(estimator, expected_failed_checks=EXPECTED_FAILED_CHECKS)

    def test_unsupported_ensemble(self):
        X, y = load_diabetes(return_X_y=True)
        estimator = AwHistGBRegressor(max_iter=5).fit(X, y)
        with self.assertRaises(TypeError):
            estimator.predict(X, weights=np.ones(X.shape[1]))


def get_datasets():
    X_bin, y_bin = load_breast_cancer(return_X_y=True)
    X_multi, y_multi = load_iris(return_X_y=True)
    y_multi = np.array(["setosa", "versicolor", "virginica"])[y_multi]
    X_reg, y_reg = load_diabetes(return_X_y=True)
    return {
        "GB_binary": (X_bin, y_bin, AwGBClassifierWithCalculator),
        "GB_multiclass": (X_multi, y_multi, AwGBClassifierWithCalculator),
        "GB_regression": (X_reg, y_reg, AwGBRegressorWithCalculator),
        "AdaBoost_binary": (X_bin, y_bin, AwAdaBoostClassifierWithCalculator),
        "AdaBoost_multiclass": (X_multi, y_multi, AwAdaBoostClassifierWithCalculator),
        "AdaBoost_regression": (X_reg, y_reg, AwAdaBoostRegressorWithCalculator),
    }

def n_stages(estimator):
    return len(estimator.estimators_)

def trees_per_stage(estimator):
    estimators = estimator.estimators_
    return estimators.shape[1] if isinstance(estimators, np.ndarray) else 1

def stage_to_tree_weights(estimator, stage_weights):
    """Repeats per-stage weights for every tree of the stage."""
    return np.repeat(stage_weights, trees_per_stage(estimator), axis=1)

def prediction_methods(estimator):
    """Names of the methods taking `weights`, with their staged counterparts."""
    names = ["predict", "decision_function", "predict_proba"]
    return [(name, f"staged_{name}") for name in names if hasattr(estimator, name)]

class WeightedBoostingPredictionTest(unittest.TestCase):

    def fit(self, X, y, estimator_class, calculator):
        return estimator_class(
            n_estimators=N_ESTIMATORS,
            random_state=42,
            tree_weight_calculator=calculator,
        ).fit(X, y)

    def assert_outputs_equal(self, actual, expected):
        if actual.dtype.kind in "fc":
            np.testing.assert_allclose(actual, expected, atol=1e-12)
        else:
            np.testing.assert_array_equal(actual, expected)

    def test_unit_tree_weights_match_unweighted(self):
        # With every tree weight 1 the original combination rule is unchanged.
        for calculator_name, calculator_class in CALCULATORS.items():
            for name, (X, y, estimator_class) in get_datasets().items():
                with self.subTest(calculator=calculator_name, dataset=name):
                    estimator = self.fit(X, y, estimator_class, calculator_class())
                    weights = np.ones(X.shape[1])
                    for method, _ in prediction_methods(estimator):
                        self.assert_outputs_equal(
                            getattr(estimator, method)(X, weights=weights),
                            getattr(estimator, method)(X),
                        )

    def test_stage_prefix_matches_staged_predictions(self):
        # Weight 1 for the first m stages and 0 for the rest is the ensemble
        # truncated after m stages, i.e. the m-th staged prediction. Each
        # sample gets its own m.
        rng = np.random.default_rng(42)
        for name, (X, y, estimator_class) in get_datasets().items():
            estimator = self.fit(X, y, estimator_class, None)
            stages = rng.integers(1, n_stages(estimator) + 1, size=len(X))
            stage_weights = (np.arange(n_stages(estimator)) < stages[:, None]).astype(float)
            estimator.tree_weight_calculator = FixedTreeWeightCalculator(
                stage_to_tree_weights(estimator, stage_weights)
            )
            for method, staged_method in prediction_methods(estimator):
                with self.subTest(dataset=name, method=method):
                    staged = np.stack(list(getattr(estimator, staged_method)(X)))
                    actual = getattr(estimator, method)(X, weights=np.ones(X.shape[1]))
                    # Staged gradient boosting decisions keep a trailing axis
                    # of size 1 in binary classification.
                    expected = staged[stages - 1, np.arange(len(X))].reshape(actual.shape)
                    self.assert_outputs_equal(actual, expected)

    def random_attribute_weights(self, rng, n_features):
        for repetition in range(3):
            yield f"positive_{repetition}", rng.uniform(0.1, 1.0, n_features)

    def stage_contributions(self, estimator, X):
        """Unnormalized contribution of each stage to the decision function.

        Derived from the staged decision function, so it follows the
        combination rule of the installed scikit-learn.
        """
        staged = np.stack(list(estimator.staged_decision_function(X)))
        if isinstance(estimator, AdaBoostClassifier):
            # The staged decision is normalized by the summed estimator weights.
            norms = np.cumsum(estimator.estimator_weights_[: n_stages(estimator)])
            staged = staged * norms.reshape((-1,) + (1,) * (staged.ndim - 1))
        return np.diff(staged, axis=0, prepend=0)

    def expected_decision(self, estimator, X, tree_weights):
        X32 = X.astype(np.float32)
        if isinstance(estimator, GradientBoostingClassifier):
            # raw = init + learning_rate * sum of w * tree output.
            raw = estimator.decision_function(X).reshape(len(X), -1).copy()
            for stage, trees in enumerate(estimator.estimators_):
                for k, tree in enumerate(trees):
                    t = stage * trees_per_stage(estimator) + k
                    raw[:, k] += estimator.learning_rate * (tree_weights[:, t] - 1) * tree.predict(X32)
            return raw.ravel() if raw.shape[1] == 1 else raw

        contributions = self.stage_contributions(estimator, X)
        alphas = estimator.estimator_weights_[: n_stages(estimator)] * tree_weights
        weighted = np.einsum("ts...,st->s...", contributions, tree_weights)
        return weighted / (alphas.sum(axis=1).reshape((-1,) + (1,) * (weighted.ndim - 1)))

    def expected_median(self, estimator, X, tree_weights):
        predictions = np.array([tree.predict(X) for tree in estimator.estimators_]).T
        alphas = estimator.estimator_weights_[: n_stages(estimator)] * tree_weights
        expected = np.empty(len(X))
        for i in range(len(X)):
            order = np.argsort(predictions[i])
            cdf = np.cumsum(alphas[i, order])
            expected[i] = predictions[i, order[np.argmax(cdf >= cdf[-1] / 2)]]
        return expected

    def test_decision_path_weights_scale_tree_weights(self):
        rng = np.random.default_rng(42)
        for calculator_name in ("geometric_mean", "arithmetic_mean"):
            for name, (X, y, estimator_class) in get_datasets().items():
                estimator = self.fit(X, y, estimator_class, CALCULATORS[calculator_name]())
                for case, attribute_weights in self.random_attribute_weights(rng, X.shape[1]):
                    with self.subTest(calculator=calculator_name, dataset=name, case=case):
                        tree_weights = estimator.tree_weight_calculator(estimator, X, attribute_weights)
                        predictions = estimator.predict(X, weights=attribute_weights)

                        if isinstance(estimator, GradientBoostingRegressor):
                            # Regression reuses the classifier's raw formula.
                            raw = estimator.predict(X).copy()
                            for t, tree in enumerate(estimator.estimators_[:, 0]):
                                raw += estimator.learning_rate * (tree_weights[:, t] - 1) * tree.predict(X.astype(np.float32))
                            np.testing.assert_allclose(predictions, raw)
                            continue
                        if isinstance(estimator, AdaBoostRegressor):
                            np.testing.assert_allclose(
                                predictions, self.expected_median(estimator, X, tree_weights)
                            )
                            continue

                        decision = estimator.decision_function(X, weights=attribute_weights)
                        expected = self.expected_decision(estimator, X, tree_weights)
                        np.testing.assert_allclose(decision, expected)

                        proba = estimator.predict_proba(X, weights=attribute_weights)
                        if isinstance(estimator, GradientBoostingClassifier):
                            expected_proba = (
                                np.c_[1 - expit(expected), expit(expected)]
                                if expected.ndim == 1 else softmax(expected, axis=1)
                            )
                        else:
                            expected_proba = estimator._compute_proba_from_decision(
                                expected.copy(), estimator.n_classes_
                            )
                        np.testing.assert_allclose(proba, expected_proba)

                        encoded = (expected > 0) if expected.ndim == 1 else np.argmax(expected, axis=1)
                        np.testing.assert_array_equal(predictions, estimator.classes_[encoded.astype(int)])
