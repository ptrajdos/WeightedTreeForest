from sklearn.utils.metaestimators import available_if

def _parent_has_method(method_name):
    """Check if the parent class has the method."""
    def check(self):
        attr = getattr(super(AttributeWeightRelatedForestMixin, self), method_name, None)
        return attr is not None
    return check

class AttributeWeightRelatedForestMixin:
    
    def _predict(self, X, weights=None, **params):
        #TODO implement
        #TODO check classification, regression
        return super().predict(X)

    def _predict_proba(self,X,weights=None, **params):
        #TODO implement
        return super().predict_proba(X, **params)

    @available_if(_parent_has_method("predict"))
    def predict(self, X, **params):
        return self._predict(X, **params)

    @available_if(_parent_has_method("predict_proba"))
    def predict_proba(self, X, **params):
        return self._predict_proba(X, **params)