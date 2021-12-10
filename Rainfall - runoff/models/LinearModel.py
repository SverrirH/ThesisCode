from ModelInterface import ModelInterface
from sklearn import LinearRegression

class LinearModel(ModelInterface):
    
    def __init__(self):
        self.model = LinearRegression()
    
    def setParameters(self, parameters: dict):
        '''Sets the model parameters'''
    
    def predict(self, X):
        '''Predicts using learned model parameters'''
        return self.model.predict(X)
    
    def fit(self, X_train, Y_train):
        '''Fits the model'''
        self.model.fit(X_train,Y_train)
        
        
    