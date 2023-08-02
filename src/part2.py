import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.2

iris = load_iris()

# X = features, y = target
iris_X = iris.data
iris_y = iris.target

dataset_size: int = len(iris_X)

X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size = TEST_SIZE)

iris_classifier = KNeighborsClassifier(n_neighbors=3)
iris_classifier.fit(X_train, y_train)

# Test
y_predict = iris_classifier.predict(X_test)
print(y_predict)
print(y_test)