from sklearn.ensemble import RandomForestClassifier


class RandomForest():
    """
    Class for the Random Forest classifier. 
    
    A wrapper for the sklearn implementation.
    """
    
    def __init__(self, max_depth=20, n_estimators=100):
        self.model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)    
    
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)
