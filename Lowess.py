import numpy as np
from sklearn import linear_model
from scipy.spatial.distance import cdist

class Lowess:
    '''Locally weighted prediction using cdist.'''
   
    def __init__(self, kernel, tau=0.5):
        # initializing lowess model
        self.kernel = kernel
        self.tau = tau
    
    def fit(self, xtrain, ytrain):
        # fitting the model on training data
        self.xtrain = xtrain
        self.ytrain = ytrain
    
    def dist(self, xtest):
        # calculating distances between training and test data
        return cdist(self.xtrain, xtest, metric='euclidean')
    
    def calculate_weights(self, calculated_distance):
        # calculating weights based on distances using the specified kernel function
        return self.kernel(calculated_distance / (2 * self.tau))
    
    def predict(self, xtest, lm=linear_model.Ridge(alpha=0.001)):
        # predicting values for the given test data using locally weighted regression
        
        # making sure test data is a 2D array
        if xtest.ndim == 1:
            xtest = xtest.reshape(-1, 1)

        distance = self.dist(xtest) # calculating distances between each point in training and test data
        weights = self.calculate_weights(distance) # computing weights for each distance using specified kernel function
        
        ypred = []  # initializing empty list to store prediction
        
        for i in range(distance.shape[1]):  # looping over each test point
            # fitting the model to the ith column of weights scaled by the training data
            lm.fit(np.diag(weights[:, i]) @ self.xtrain, np.diag(weights[:, i]) @ self.ytrain)
            # predicting the value for the ith test point and reshaping for correct dimensions
            ypred.append(lm.predict(xtest[i].reshape(1, -1)))
            
        return np.array(ypred).flatten() #converting the list of predictions into a numpy array and flattening for proper output