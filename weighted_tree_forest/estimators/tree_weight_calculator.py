from abc import ABC, abstractmethod

import numpy as np
from scipy import sparse
from sklearn.utils.validation import _num_samples


class TreeWeightCalculator(ABC):
    """Interface for objects that turn attribute weights into tree weights.

    Implementations receive a fitted forest, the samples being predicted and a
    vector of attribute (feature) weights, and return one weight per sample and
    tree in ``forest.estimators_``. All weights lie in the [0, 1] interval.
    """

    @abstractmethod
    def calculate_tree_weights(self, forest, X, attribute_weights):
        """Calculate tree weights.

        Parameters
        ----------
        forest : fitted forest estimator
            Forest exposing ``estimators_`` and ``n_features_in_``.
        X : array-like of shape (n_samples, n_features)
            Samples being predicted.
        attribute_weights : ndarray of shape (n_features,)
            Weight of each attribute, in [0, 1].

        Returns
        -------
        tree_weights : ndarray of shape (n_samples, n_estimators)
            Weight of each tree for each sample, in [0, 1].
        """

    def __call__(self, forest, X, attribute_weights):
        attribute_weights = np.asarray(attribute_weights, dtype=float)
        if attribute_weights.shape != (forest.n_features_in_,):
            raise ValueError(
                f"attribute_weights must have shape ({forest.n_features_in_},), "
                f"got {attribute_weights.shape}."
            )
        if np.any((attribute_weights < 0) | (attribute_weights > 1)):
            raise ValueError("attribute_weights must lie in the [0, 1] interval.")

        tree_weights = np.asarray(
            self.calculate_tree_weights(forest, X, attribute_weights), dtype=float
        )
        expected_shape = (_num_samples(X), len(forest.estimators_))
        if tree_weights.shape != expected_shape:
            raise ValueError(
                f"Tree weights must have shape {expected_shape}, "
                f"got {tree_weights.shape}."
            )
        if np.any((tree_weights < 0) | (tree_weights > 1)):
            raise ValueError("Tree weights must lie in the [0, 1] interval.")
        return tree_weights


class UniformTreeWeightCalculator(TreeWeightCalculator):
    """Assigns equal weight to every tree, ignoring attribute weights."""

    def calculate_tree_weights(self, forest, X, attribute_weights):
        return np.ones((_num_samples(X), len(forest.estimators_)))


class _DecisionPathMeanTreeWeightCalculator(TreeWeightCalculator):
    """Base for calculators averaging attribute weights along decision paths.

    An attribute tested several times on a path counts each time. A tree that
    is a single leaf tests no attributes and gets weight 1.
    """

    def _path_means(self, forest, X, node_values):
        """Mean of per-attribute values over the internal nodes on each path.

        Parameters
        ----------
        node_values : ndarray of shape (n_features,)
            Value contributed by each internal node testing the attribute.

        Returns
        -------
        mean : ndarray of shape (n_samples, n_estimators)
            Mean value on the path of each sample in each tree, 0 for paths
            without internal nodes.
        has_internal_nodes : ndarray of shape (n_samples, n_estimators)
            Whether the path tests any attribute.
        """
        # indicator[i, j] == 1 iff sample i visits node j; the nodes of tree t
        # are columns n_nodes_ptr[t]:n_nodes_ptr[t + 1].
        indicator, n_nodes_ptr = forest.decision_path(X)

        features = np.concatenate([tree.tree_.feature for tree in forest.estimators_])
        is_internal = features >= 0

        # Maps each internal node to its tree, so that indicator @ matrix sums
        # node values per tree.
        nodes = np.flatnonzero(is_internal)
        node_tree = np.searchsorted(n_nodes_ptr, nodes, side="right") - 1
        shape = (len(features), len(forest.estimators_))
        value_sum = indicator @ sparse.csr_array(
            (node_values[features[is_internal]], (nodes, node_tree)), shape=shape
        )
        path_length = indicator @ sparse.csr_array(
            (np.ones(len(nodes)), (nodes, node_tree)), shape=shape
        )

        value_sum = sparse.csr_array(value_sum).toarray()
        path_length = sparse.csr_array(path_length).toarray()
        has_internal_nodes = path_length > 0
        with np.errstate(invalid="ignore"):
            mean = np.where(
                has_internal_nodes, value_sum / np.maximum(path_length, 1), 0.0
            )
        return mean, has_internal_nodes


class DecisionPathTreeWeightCalculator(_DecisionPathMeanTreeWeightCalculator):
    """Weights each tree by the attribute weights along a sample's decision path.

    The weight of tree ``t`` for sample ``x`` is the geometric mean of the
    weights of the attributes tested in the internal nodes that ``x`` visits in
    ``t``. An attribute tested several times on the path counts each time. A
    tree that is a single leaf tests no attributes and gets weight 1.
    """

    def calculate_tree_weights(self, forest, X, attribute_weights):
        with np.errstate(divide="ignore"):
            log_weights = np.log(attribute_weights)
        mean_log_weight, _ = self._path_means(forest, X, log_weights)
        return np.exp(mean_log_weight)


class DecisionPathArithmeticMeanTreeWeightCalculator(_DecisionPathMeanTreeWeightCalculator):
    """Weights each tree by the attribute weights along a sample's decision path.

    The weight of tree ``t`` for sample ``x`` is the arithmetic mean of the
    weights of the attributes tested in the internal nodes that ``x`` visits in
    ``t``. An attribute tested several times on the path counts each time. A
    tree that is a single leaf tests no attributes and gets weight 1.
    """

    def calculate_tree_weights(self, forest, X, attribute_weights):
        mean_weight, has_internal_nodes = self._path_means(forest, X, attribute_weights)
        return np.where(has_internal_nodes, mean_weight, 1.0)
