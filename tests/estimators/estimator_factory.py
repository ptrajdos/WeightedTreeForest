import inspect


def with_tree_weight_calculator(mixin_class, estimator_class, name, module):
    """Subclass of `mixin_class` and `estimator_class` taking a tree_weight_calculator.

    The constructor signature is copied from the installed scikit-learn, so it
    follows the estimator parameters of each version (e.g. monotonic_cst since
    1.4). Assign the class to `name` in `module` to keep instances picklable.
    """
    parent_init = estimator_class.__init__
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
    return type(
        name,
        (mixin_class, estimator_class),
        {"__init__": __init__, "__module__": module, "__qualname__": name},
    )
