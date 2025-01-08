import numpy as np

class Scaler():
    def __init__(self, minn=0, maxx=1):
        self.min = minn
        self.max = maxx

    def fit_transform(self, X):
    #returns columns of X transformed to the range [min, max]
        assert type(X) is np.ndarray, "X must be a numpy array"


        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (self.max - self.min) + self.min
        self.data_min_= X.min(axis=0)
        self.data_max_= X.max(axis=0)
        self.sampleLength= X.shape[0]
        return np.array(X_scaled)
    

    def inverse_transform(self,X):
    #returns columns of X transformed back to the original scale
        assert self.data_min_ is not None , "data_min is empty"
        assert self.data_max_ is not None , "data_max is empty"
        assert type(X) is np.ndarray, "X must be a numpy array"
            
        return np.array((X - self.min) / (self.max - self.min) * (self.data_max_ - self.data_min_) + self.data_min_)
    

    def inverse_transform_column(self, X, index):
    #returns the indexed columns of X transformed back to the original scale

        assert self.data_min_ is not None , "data_min is empty"
        assert self.data_max_ is not None , "data_max is empty"
        assert type(X) is np.ndarray, "X must be a numpy array"
        #must check if data_min_ and X has the same dimentions
        # assert self.data_min_[index].shape == X.shape[1], "data_min and X must have the same number of columns"

        #there is still 2d array problem here

        return np.array((X - self.min) / (self.max - self.min) * (self.data_max_[index] - self.data_min_[index]) + self.data_min_[index])
        
# scaler=Scaler()
# X = np.array([[1, 2, 3], [4, 5, 6], [4, -3, 2]])
# tr=scaler.fit_transform(X)
# print(tr)
# print(scaler.inverse_transform(tr))
# print("-----------------")
# print(tr[:,0:2])
# print(scaler.inverse_transform_column(tr[:,0:2], [0,1]))