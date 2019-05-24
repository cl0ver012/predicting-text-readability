from xgboost import XGBClassifier


class XGBoost():
    """
    Class for the XGBoost classifier. 
    
    A wrapper for the xgboost implemenation.
    """
    
    def __init__(self, max_depth=30, n_estimators=200, learning_rate=0.1):
        self.model = xgboost = XGBClassifier(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate)  
    
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)