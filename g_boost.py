import numpy as np

class GradientBooster:
    '''Gradient Boosting model combining predictions from a main model 
    and a residual model using boosting steps to improve predictive performance.'''
    
    def __init__(self, model, rmodel, boosting_steps):
        # initializing the model
        self.model = model  # main model 
        self.rmodel = rmodel   # residual model
        self.steps = boosting_steps  # number of boosting steps
        self.is_fitted = False    # flag to check if the model has been fitted
        return
    
    def fit(self, xtrain, ytrain):
        # fitting the GradientBooster model to training data
        self.xtrain = xtrain  # storing training features
        self.ytrain = ytrain   # storing training target
        self.is_fitted = True  # changing the fitted flag to True

        # fitting main model to training data
        self.model.fit(self.xtrain, self.ytrain)
        # predicting on training data
        y_train_pred = self.model.predict(self.xtrain)  

        # completing the boosting process: iteratively fitting the residuals
        for i in range(self.steps):
            residuals = self.ytrain - y_train_pred  # calculating the residuals
            self.rmodel.fit(xtrain, residuals)  # fitting residual model to training features and residuals
            y_train_pred += self.rmodel.predict(self.xtrain)  # updating predictions with residual model predictions
        return
    
    def predict(self, xtest):

        # checking that model is fitted before predicting
        if not self.is_fitted:
            raise ValueError("is_fitted is false. Please fit the model.")

        # predicting using the main model
        ypred = self.model.predict(xtest)
        # adjusting predictions by subtracting residual model predictions
        ypred = ypred - self.rmodel.predict(xtest)

        # returning the final predictions
        return ypred