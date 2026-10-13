from sklearn.utils.validation import check_is_fitted

from weighted_tree_forest.estimators.tree_weight_calculator import (
    TreeWeightCalculator,
    UniformTreeWeightCalculator,
)


class TreeWeightingMixin:
    """Computes per-sample tree weights from attribute weights.

    Base of the forest and boosting mixins.
    """

    # Object calculating tree weights from attribute weights. Concrete
    # estimators may override it as a class attribute or expose it as an
    # ``__init__`` parameter. ``None`` means UniformTreeWeightCalculator.
    tree_weight_calculator = None

    def _get_tree_weight_calculator(self):
        calculator = self.tree_weight_calculator
        if calculator is None:
            return UniformTreeWeightCalculator()
        if not isinstance(calculator, TreeWeightCalculator):
            raise TypeError(
                "tree_weight_calculator must be a TreeWeightCalculator, "
                f"got {type(calculator).__name__}."
            )
        return calculator

    def _compute_tree_weights(self, X, attribute_weights):
        check_is_fitted(self)
        return self._get_tree_weight_calculator()(self, X, attribute_weights)
