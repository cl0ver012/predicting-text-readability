from sklearn.svm import SVC
import pickle


class SupportVectorMachine():
    """
    Class for the Support Vector Machine (SVM) classifier.
    
    A wrapper for the sklearn implementation.
    """
    
    def __init__(self, kernel='linear', C=10.0, save_model=False, use_saved_model=False, model_path='./models/saved_models/svm.pickle'):
        self.model_path = model_path
        self.save_model = save_model
        
        if use_saved_model:
            with open(self.model_path, 'rb') as file:
                self.model = pickle.load(file)
        else:
            self.model = SVC(kernel=kernel, C=C)
    
    
    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        
        if self.save_model:
            with open(self.model_path, 'wb') as handle:
                pickle.dump(self.model, handle)
    
    
    def predict(self, X_test):
        return self.model.predict(X_test)