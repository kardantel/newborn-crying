from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer
from sklearn.preprocessing import LabelEncoder
import numpy as np
from utils import Utils

utils = Utils()

class MLP:
    def __init__(self, df, random_state=None):
        self.df = df.copy()
        self.encoder = LabelEncoder()
        self.encoder.fit(self.df.iloc[:, -1])
        self.df.iloc[:, -1] = self.encoder.transform(self.df.iloc[:, -1])
        self.X_train = self.df.sample(frac=0.7, random_state=random_state)
        self.X_test = self.df.drop(self.X_train.index)
        self.y_train = self.X_train.pop(df.columns[-1])
        self.y_test = self.X_test.pop(df.columns[-1])
        self.classes = df.iloc[:, -1].unique()

        self.model = Sequential()
        self.model.add(InputLayer(input_shape=self.X_train.shape[1]))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(self.classes.shape[0], activation='softmax'))
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.fit(self.X_train, self.y_train,
                       batch_size=100, epochs=100, verbose=2)

        self.y_pred = self.model.predict(self.X_test, batch_size=100, verbose=2)
        self.y_pred_class = np.argmax(self.y_pred, axis=1)
        
        utils.getMetrics(self.y_test, self.y_pred_class)

    def predict(self):
        print(f'RFs Prediction: {self.model.predict(self.X_test, batch_size=1)}')
