from math import gamma
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.utils.fixes import loguniform
from utils import Utils

utils = Utils()

class SVM:
    def __init__(self, df, random_state=None):
        self.df = df
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        self.classes = self.y.unique()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.7, random_state=random_state)

        self.dist = dict(C=loguniform(1e-3, 1e3),
                         gamma=loguniform(1e-6, 1e-2),
                         kernel=['rbf', 'linear', 'sigmoid'])
        self.rfc = SVC()
        self.clf = RandomizedSearchCV(estimator=self.rfc,
                                      param_distributions=self.dist,
                                      n_iter=10,
                                      scoring='balanced_accuracy',
                                      n_jobs=-1,
                                      verbose=3).fit(self.X_train, self.y_train)

        self.y_pred = self.clf.predict(self.X_test)
        
        utils.getMetrics(self.y_test, self.y_pred)

    def predict(self):
        print(f'RFs Prediction: {self.clf.predict(self.X_test)}')
