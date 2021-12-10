class ModelInterface:
    
    def getParameters(self):
        '''Returns a dictionary with the model parameters'''
        raise Exception('This function has not been implemented')
        
    def setParameters(self, parameters: dict):
        '''Sets the model parameters'''
        raise Exception('This function has not been implemented')
    
    def predict(self, X):
        '''Predicts using learned model parameters'''
        raise Exception('This function has not been implemented')
    
    def fit(self, X_train, Y_train, logger)):
        '''Fits the model and logs performance during and/or after training'''
        raise Exception('This function has not been implemented')
        
        
    