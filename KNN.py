import numpy as np
from collections import Counter

def euclidean_distance(x1,x2):
    return np.sqrt(np.sum((x1-x2)**2))

class KNN:
    def __init__(self,k=3):
        self.k = k

    def fit(self,X ,y):
        self.X_train=X
        self.y_train=y

    def predict(self,X):
        return [self._predict(x) for x in X]
    
    def _predict(self,x):
        # calculate the distances 
        distances = [euclidean_distance(x,x_train) for x_train in self.X_train]

        #sort in acsending order and pick up the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        neighbors = [self.y_train[i] for i in k_indices]

        # return most common label
        print(Counter(neighbors).most_common())
