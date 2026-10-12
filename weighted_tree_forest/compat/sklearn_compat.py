import inspect
import warnings
from unittest import SkipTest

from sklearn.exceptions import SkipTestWarning
from sklearn.utils.estimator_checks import check_estimator

try:  # sklearn >= 1.6
    from sklearn.utils.validation import _check_n_features
    def check_n_features_internal(estimator, X, reset):
        _check_n_features(estimator=estimator, X=X, reset=reset)
except ImportError:  # sklearn < 1.6
    def check_n_features_internal(estimator, X, reset):
        estimator._check_n_features(X, reset)

try:  # sklearn >= 1.6
    from sklearn.utils.validation import validate_data
except ImportError:  # sklearn < 1.6
    def validate_data(estimator, X="no_validation", y="no_validation", **kwargs):
        return estimator._validate_data(X, y, **kwargs)


def validate_fit_external(estimator, X, y):
    X, y = validate_data(
        estimator,
        X=X,
        y=y,
        accept_sparse=False,
        order="C",
        accept_large_sparse=False,
        y_numeric=False,
        reset=True,
    )
    return X, y


def validate_predict_input_ext(estimator, X):
    X = validate_data(
        estimator,
        X=X,
        accept_sparse=False,
        order="C",
        accept_large_sparse=False,
        reset=False,
    )
    return X


if "expected_failed_checks" in inspect.signature(check_estimator).parameters:  # sklearn >= 1.6
    def check_estimator_compat(estimator, expected_failed_checks=None):
        check_estimator(estimator, expected_failed_checks=expected_failed_checks)
else:  # sklearn < 1.6
    def check_estimator_compat(estimator, expected_failed_checks=None):
        # expected_failed_checks is not supported; skip those checks instead.
        expected_failed_checks = expected_failed_checks or {}
        for est, check in check_estimator(estimator, generate_only=True):
            check_name = getattr(check, "func", check).__name__
            if check_name in expected_failed_checks:
                continue
            # Like check_estimator: a skipped check must not stop the others.
            try:
                check(est)
            except SkipTest as e:
                warnings.warn(
                    f"Skipping {check_name} for {type(est).__name__}: {e}",
                    SkipTestWarning,
                )
