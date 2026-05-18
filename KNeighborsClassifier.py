import numpy as np
from collections import Counter
class Knn:
    def __init__(self,k=5):
        self.n_neighbors = k
        self.x_train = None
        self.y_train = None

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self,x_test):
        y_pred = []
        for i in x_test: #Calculate distance of each point.
            distances = []
            for j in self.x_train:
                distances.append(self.calculate_distance(i,j)) #Append distance.
            n_neighbors= sorted(list(enumerate(distances)),key= lambda x:x[1])[0:self.n_neighbors] #when sorting, use the distance value to decide the order in key= lambda x:x[1].
            label = self.majority_count(n_neighbors) #Get label of neighbors.
            y_pred.append(label)
        return np.array(y_pred)

    def calculate_distance(self,point_A,point_B):
       return np.linalg.norm(point_A - point_B)
    
    def majority_count(self,neighbors):
        votes = []
        for i in neighbors:
            votes.append(self.y_train.iloc[i[0]]) #Mask the index of neighbors on y_train.
        votes = Counter(votes)
        return votes.most_common()[0][0]