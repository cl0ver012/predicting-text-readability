from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np


class MultilayerPerceptron():
    """
    Class for the Multilayer Perceptron (MLP) model.
    Implemented in Keras.
    """
    
    def __init__(self, input_dim):
        self.model = self._make_model(input_dim)
    
    
    def fit(self, X_train, y_train):
        y_train_cat = to_categorical(y_train, num_classes=5)
        
        self.model.fit(
            X_train, y_train_cat,
            epochs=150,
            batch_size=64,
            validation_split=0.2,
            verbose=1)
    
    
    def predict(self, X_test): 
        y_pred_cat = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_cat, axis=1)
        
        return y_pred_cat
    
    
    def _make_model(self, input_dim):
        
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(5, activation='softmax'))
        
        adam = Adam(lr=0.001)
        model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        
        return model
        