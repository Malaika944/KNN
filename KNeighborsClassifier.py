class Knn:
    def __init__(self,k=5):
        self.n_neighbors = k
        self.x_train = None
        self.y_train = None

    def fit(self,x_train,y_train):
        self.x_train = x_train
        self.y_train = y_train
    
    def predict(self,x_test):
        for i in x_test:
            print(i)