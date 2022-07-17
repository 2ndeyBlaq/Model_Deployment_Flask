# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Read dataset
iris = pd.read_csv("Iris.csv")
print(iris.columns)         # Print all columns of dataset
print(iris.head())          # Print top 5 rows

# Define model features y: Target variable
y = iris['species']
iris.drop(columns='species',inplace=True)
X = iris[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
print('y: ', y.shape, 'X: ', X.shape)

# Training the model
x_train,x_test,y_train,y_test = train_test_split(X,y, test_size=0.3)
model = LogisticRegression()
model.fit(x_train,y_train)

# Print model accuracy
print(model.score(x_train, y_train))

# Save the model
pickle.dump(model, open('model.pkl','wb'))