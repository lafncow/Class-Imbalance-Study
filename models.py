# ## Load Libraries

from sklearn.linear_model import LogisticRegression


class logistic:
    def __init__(self):
        self.name = "Logistic"
        self.model = LogisticRegression()
        return self
    
    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)
    
    def pred(self, x):
        self.model.predict(x)
