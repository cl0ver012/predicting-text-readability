from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
import numpy as np
import tensorflow as tf

from .utils.utils import discretize

class MultilayerPerceptron():
    """
    Class for the Multilayer Perceptron (MLP) model.
    Implemented in Keras.
    """
    
    def __init__(self, input_dim=None, verbose=0, save_model=False, use_saved_model=False, model_path='./models/saved_models/mlp.h5'):
        self.model_path = model_path
        self.save_model = save_model
        
        self.input_dim = input_dim
        self.verbose = verbose

        # do not write warnings
        tf.logging.set_verbosity(tf.logging.ERROR)
        
        if use_saved_model:
            self.model = load_model(model_path)
    
    
    def fit(self, X_train, y_train):
        self.model = self._make_model()
        
        y_train_cat = to_categorical(y_train, num_classes=5)
        
        self.model.fit(
            X_train, y_train_cat,
            epochs=150,
            batch_size=64,
            verbose=self.verbose)
        
        if self.save_model:
            self.model.save(self.model_path)
    
    
    def predict(self, X_test): 
        y_pred_cat = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_cat, axis=1)
        
        return discretize(y_pred)
    
    
    def _make_model(self):
        
        # architecture
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=self.input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(5, activation='linear'))
        
        # opitimizer
        adam = Adam(lr=0.001)
        
        model.compile(optimizer=adam,
              loss='mse',
              metrics=['mse'])
        
        return model
        