import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from KNeighborsClassifier import Knn
 
df = pd.read_csv('data/raw/Social_Network_Ads.csv')
print(df)
#Encode categorical variable.
df['Gender'] = encoder.fit_transform(df['Gender'])
print(df)
#Split and transform data.
x= df.iloc[:,0:3].values
x= scaler.fit_transform(x)
y = df.iloc[:,-1]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 0)
#Use sklearn's KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)
print("accuracy:",accuracy_score(y_test,y_pred))
#My own Knn.
myknn = Knn(k=5)
myknn.fit(x_train,y_train)
myknn.predict(x_test)