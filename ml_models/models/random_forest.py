from sklearn.ensemble import RandomForestClassifier
import pickle


class RandomForest():
    """
    Class for the Random Forest classifier. 
    
    A wrapper for the sklearn implementation.
    """
    
    def __init__(self, max_depth=20, n_estimators=100, save_model=False, use_saved_model=False, model_path='./models/saved_models/rf.pickle'):
        self.model_path = model_path
        self.save_model = save_model
        
        if use_saved_model:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            self.model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)    
    
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
        if self.save_model:
            with open(self.model_path, 'wb') as handle:
                pickle.dump(self.model, handle)
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)
