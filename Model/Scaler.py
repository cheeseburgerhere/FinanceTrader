import numpy as np

class Scaler():
    def __init__(self, minn=0, maxx=1):
        self.min = minn
        self.max = maxx

    def fit_transform(self, X):
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (self.max - self.min) + self.min
        self.data_min_= X.min(axis=0)
        self.data_max_= X.max(axis=0)
        self.sampleLength= X.shape[0]
        return X_scaled
    

    def inverse_transform(self,X):
        #returns columns of X transformed back to the original scale
        assert self.data_min_ is not None , "data_min is empty"
        assert self.data_max_ is not None , "data_max is empty"
            
        return (X - self.min) / (self.max - self.min) * (self.data_max_ - self.data_min_) + self.data_min_
    

    def inverse_transform_column(self, X, index):
        #returns the indexed columns of X transformed back to the original scale

        assert self.data_min_ is not None , "data_min is empty"
        assert self.data_max_ is not None , "data_max is empty"
        
        return (X[:, index] - self.min) / (self.max - self.min) * (self.data_max_[index] - self.data_min_[index]) + self.data_min_[index]
        
# scaler=Scaler()
# X = np.array([[1, 2, 3], [4, 5, 6], [4, 3, 2]])
# tr=scaler.fit_transform(X)
# print(tr)
# print(scaler.inverse_transform(tr))
# print(scaler.inverse_transform_column(tr, [0,1]))