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

# Get the iris dataset (dictionary)
iris = load_iris()

# https://stackoverflow.com/questions/32137396/how-do-i-plot-only-a-table-in-matplotlib
num_rows = len(iris.data)
classified_data = []

# Add  classification name column
for i in range(len(iris.data)):
    # The data is stored as a numpy array
    row = iris.data[i].tolist()
    # Find the id of the classification name, search for the classification name and add to row
    classification_id = iris.target[i]
    row.append(iris.target_names[classification_id])
    classified_data.append(row)

iris_dataframe = pd.DataFrame(data=classified_data, columns=['Sepal length (cm)', 'Sepal width (cm)', 'Petal length (cm)', 'Petal width (cm)', 'Classification name'])
print(iris.DESCR)

display(iris_dataframe)

# plt.axis('off')
# plt.axis('tight')
# plt.tight_layout()

# # Plot a data table and hide main graph
# table = plt.table(cellText=iris_dataframe.values, colLabels=iris_dataframe.columns, loc='best')

# plt.show()