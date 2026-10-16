import copy

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
)
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from weighted_tree_forest.compat.sklearn_compat import (
    gradient_boosting_raw_to_decision,
    gradient_boosting_raw_to_proba,
    validate_data,
)
from weighted_tree_forest.estimators.tree_weighting_mixin import TreeWeightingMixin

def _parent_has_method(method_name):
    """Check if the parent class has the method."""
    def check(self):
        attr = getattr(super(AttributeWeightRelatedBoostingMixin, self), method_name, None)
        return attr is not None
    return check

class AttributeWeightRelatedBoostingMixin(TreeWeightingMixin):
    """Attribute-weighted predictions for scikit-learn boosting ensembles.

    Mix it in before ``GradientBoostingClassifier``,
    ``GradientBoostingRegressor``, ``AdaBoostClassifier`` or
    ``AdaBoostRegressor`` (with decision tree base estimators). Prediction
    methods accept ``weights``, the attribute weights. The tree weight
    calculator turns them into per-sample tree weights ``w``, which multiply
    the weights the ensemble gives its trees; the ensemble's own combination
    rule is kept:

    - gradient boosting: the init prediction plus ``learning_rate * w`` times
      each tree output, converted by the loss as usual;
    - AdaBoost classifier: the SAMME (or, before scikit-learn 1.6, SAMME.R)
      vote of the installed scikit-learn, with estimator weights
      ``estimator_weights_ * w`` normalized by their sum;
    - AdaBoost regressor: the weighted median with weights
      ``estimator_weights_ * w``.

    With all tree weights equal to 1 the predictions equal the unweighted
    ones. Staged prediction methods are not weighted.
    """

    def _gradient_boosting_raw_predict(self, X, weights):
        check_is_fitted(self)
        X = validate_data(
            self, X, dtype=np.float32, order="C", accept_sparse="csr", reset=False
        )
        n_stages, n_trees_per_stage = self.estimators_.shape
        tree_weights = self._compute_tree_weights(X, weights).reshape(
            -1, n_stages, n_trees_per_stage
        )
        raw_predictions = self._raw_predict_init(X)
        for stage, trees in enumerate(self.estimators_):
            for k, tree in enumerate(trees):
                raw_predictions[:, k] += (
                    self.learning_rate
                    * tree_weights[:, stage, k]
                    * tree.predict(X, check_input=False)
                )
        return raw_predictions

    def _adaboost_estimator_weights(self, X, weights):
        """Estimator weights of shape (n_samples, n_estimators), scaled by tree weights."""
        n_estimators = len(self.estimators_)
        return self.estimator_weights_[:n_estimators] * self._compute_tree_weights(
            X, weights
        )

    def _adaboost_decision_function(self, X, weights):
        check_is_fitted(self)
        estimator_weights = self._adaboost_estimator_weights(self._check_X(X), weights)

        # The vote of each estimator is the ensemble's own decision function
        # on a copy holding only that estimator, so the voting rule of the
        # installed scikit-learn is used. Votes are linear in the estimator
        # weights, so their weighted mean is the weighted ensemble vote.
        single = copy.copy(self)
        single.estimator_weights_ = np.ones(1)
        decision = 0.0
        for estimator, weight in zip(self.estimators_, estimator_weights.T):
            single.estimators_ = [estimator]
            vote = super(AttributeWeightRelatedBoostingMixin, single).decision_function(X)
            decision = decision + weight.reshape((-1,) + (1,) * (vote.ndim - 1)) * vote

        norm = estimator_weights.sum(axis=1).reshape((-1,) + (1,) * (decision.ndim - 1))
        # Samples whose trees all have weight 0 get a neutral decision.
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(norm > 0, decision / norm, 0.0)

    def _adaboost_median_predict(self, X, weights):
        check_is_fitted(self)
        X = self._check_X(X)
        estimator_weights = self._adaboost_estimator_weights(X, weights)
        # The median ignores a common scale of the weights, so samples whose
        # trees all have weight 0 keep the unweighted median.
        unweighted = estimator_weights.sum(axis=1) == 0
        estimator_weights[unweighted] = self.estimator_weights_[: len(self.estimators_)]

        predictions =np.array([estimator.predict(X) for estimator in self.estimators_]).T
        sorted_idx = np.argsort(predictions, axis=1)
        weight_cdf = np.cumsum(
            np.take_along_axis(estimator_weights, sorted_idx, axis=1), axis=1
        )
        median_or_above = weight_cdf >= 0.5 * weight_cdf[:, -1:]
        median_idx = median_or_above.argmax(axis=1)
        median_estimators = np.take_along_axis(sorted_idx, median_idx[:, None], axis=1)
        return np.take_along_axis(predictions, median_estimators, axis=1).ravel()

    def _weighted_decision_function(self, X, weights):
        if isinstance(self, GradientBoostingClassifier):
            raw_predictions = self._gradient_boosting_raw_predict(X, weights)
            if raw_predictions.shape[1] == 1:
                return raw_predictions.ravel()
            return raw_predictions
        if isinstance(self, AdaBoostClassifier):
            return self._adaboost_decision_function(X, weights)
        raise self._unsupported_error()

    def _unsupported_error(self):
        return TypeError(
            f"{type(self).__name__} is not a GradientBoostingClassifier, "
            "GradientBoostingRegressor, AdaBoostClassifier or AdaBoostRegressor."
        )

    def predict(self, X, weights=None):
        if weights is None:
            return super().predict(X)

        if isinstance(self, GradientBoostingRegressor):
            return self._gradient_boosting_raw_predict(X, weights).ravel()
        if isinstance(self, AdaBoostRegressor):
            return self._adaboost_median_predict(X, weights)

        decision = self._weighted_decision_function(X, weights)
        if isinstance(self, GradientBoostingClassifier):
            encoded_labels = gradient_boosting_raw_to_decision(self, decision)
            return self.classes_.take(encoded_labels, axis=0)
        if self.n_classes_ == 2:
            return self.classes_.take(decision > 0, axis=0)
        return self.classes_.take(np.argmax(decision, axis=1), axis=0)

    @available_if(_parent_has_method("decision_function"))
    def decision_function(self, X, weights=None):
        if weights is None:
            return super().decision_function(X)
        return self._weighted_decision_function(X, weights)

    @available_if(_parent_has_method("predict_proba"))
    def predict_proba(self, X, weights=None):
        if weights is None:
            return super().predict_proba(X)

        decision = self._weighted_decision_function(X, weights)
        if isinstance(self, GradientBoostingClassifier):
            return gradient_boosting_raw_to_proba(self, decision)
        if self.n_classes_ == 1:
            return np.ones((decision.shape[0], 1))
        return self._compute_proba_from_decision(decision, self.n_classes_)
