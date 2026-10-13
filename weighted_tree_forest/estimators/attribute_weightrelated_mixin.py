import numpy as np
from sklearn.base import is_classifier
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted

from weighted_tree_forest.estimators.tree_weighting_mixin import TreeWeightingMixin

def _parent_has_method(method_name):
    """Check if the parent class has the method."""
    def check(self):
        attr = getattr(super(AttributeWeightRelatedForestMixin, self), method_name, None)
        return attr is not None
    return check

class AttributeWeightRelatedForestMixin(TreeWeightingMixin):

    def _weighted_tree_sum(self, tree_outputs, tree_weights):
        """Sum per-tree outputs, each of shape (n_samples, ...), weighted by tree."""
        result = None
        for tree_output, weight in zip(tree_outputs, tree_weights.T):
            weight = weight.reshape((-1,) + (1,) * (tree_output.ndim - 1))
            contribution = weight * tree_output
            result = contribution if result is None else result + contribution
        return result

    def _predict(self, X, weights=None, **params):
        if weights is None:
            return super().predict(X)

        if not is_classifier(self):
            check_is_fitted(self)
            X = self._validate_X_predict(X)
            tree_weights = self._compute_tree_weights(X, weights)
            return self._weighted_tree_sum(
                (tree.predict(X, check_input=False) for tree in self.estimators_),
                tree_weights,
            )

        # Weighted soft voting, as in the unweighted random forest classifier.
        proba = self._predict_proba(X, weights=weights)
        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        predictions = np.empty((proba[0].shape[0], self.n_outputs_), dtype=self.classes_[0].dtype)
        for k in range(self.n_outputs_):
            predictions[:, k] = self.classes_[k].take(np.argmax(proba[k], axis=1), axis=0)
        return predictions

    def _predict_proba(self, X, weights=None, **params):
        if weights is None:
            return super().predict_proba(X, **params)

        check_is_fitted(self)
        X = self._validate_X_predict(X)
        tree_weights = self._compute_tree_weights(X, weights)
        tree_probas = [tree.predict_proba(X, check_input=False) for tree in self.estimators_]
        if self.n_outputs_ == 1:
            return self._weighted_tree_sum(tree_probas, tree_weights)
        return [
            self._weighted_tree_sum((proba[k] for proba in tree_probas), tree_weights)
            for k in range(self.n_outputs_)
        ]

    @available_if(_parent_has_method("predict"))
    def predict(self, X, **params):
        return self._predict(X, **params)

    @available_if(_parent_has_method("predict_proba"))
    def predict_proba(self, X, **params):
        return self._predict_proba(X, **params)
