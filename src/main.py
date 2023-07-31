import pandas as pd
3 import seaborn as sns
4 import matplotlib.pyplot as plt
5 from sklearn.datasets import load_iris
6 from sklearn.model_selection import train_test_split
7 from sklearn.neighbors import KNeighborsClassifier
8 from sklearn.metrics import accuracy_score
9 from sklearn.metrics import plot_confusion_matrix