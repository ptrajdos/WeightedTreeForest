import unittest
from sklearn.utils.estimator_checks import check_estimator
from weighted_tree_forest.estimators.attribute_weightrelated_mixin import AttributeWeightRelatedForestMixin

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

class AwRFCclass(AttributeWeightRelatedForestMixin, RandomForestClassifier):
    pass

class AWRfRegressor(AttributeWeightRelatedForestMixin, RandomForestRegressor):
    pass

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

class AttributeWeightRelatedForestMixinTest(unittest.TestCase):
    
    def get_estimators(self):
        return {
            "RFClassifier": AwRFCclass(n_estimators=10, random_state=42),
            "RFRegressor": AWRfRegressor(n_estimators=10, random_state=42)
        
        }   

    def test_sklearn(self):

        for name, clf in self.get_estimators().items():
            with self.subTest(estimator=name):
                check_estimator(clf, expected_failed_checks=EXPECTED_FAILED_CHECKS)