from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from utils import Utils

utils = Utils()

class RF:
    def __init__(self, df, random_state=None):
        self.df = df
        self.X = df.iloc[:, :-1]
        self.y = df.iloc[:, -1]
        self.classes = self.y.unique()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.7, random_state=random_state)

        self.dist = dict(max_features=[*range(10)], max_depth=[*range(30)])
        self.rfc = RandomForestClassifier(n_estimators=100,
                                          max_features='auto',
                                          n_jobs=-1, random_state=random_state)
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
