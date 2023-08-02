from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

TEST_SIZE = 0.2

iris = load_iris()

# X represents features and y represents target (classification)
iris_X = iris.data
iris_y = iris.target

def test_nearest_neighbours(k: int, weights: str):
    print("New nearest neighbours model with k=" + str(k) + " and " + weights + " weighting")

    X_train, X_test, y_train, y_true = train_test_split(iris_X, iris_y, test_size = TEST_SIZE)
    iris_classifier = KNeighborsClassifier(n_neighbors=k, weights=weights)
    iris_classifier.fit(X_train, y_train)
    y_pred = iris_classifier.predict(X_test)
    conf_matrix = confusion_matrix(y_true, y_pred)
    print(conf_matrix)

    # Count successes and failure from matrix
    success_count = 0
    fail_count = 0
    for i in range(len(conf_matrix)):
        row = conf_matrix[i]
        for j in range(len(row)):
            # If it is a diagonal, add to successes, else add to fail
            if i == j:
                success_count += conf_matrix[i][j]
            else:
                fail_count += conf_matrix[i][j]

    print("Success: " + str(success_count) + ", failures: " + str(fail_count))
    print("Accuracy: " + str(accuracy_score(y_true, y_pred)))
    print("\n")

# Confusion matrix for different k-values and uniform weights
print("Uniform weighting nearest neighbors \n")
for k in range(1, 21):
    test_nearest_neighbours(k, 'uniform')

# Confusion matrix for different k-values and uniform weights
print("Inverse distance weighting nearest neighbors \n")
for k in range(1, 21):
    test_nearest_neighbours(k, 'distance')

# Reflective questions
# 1. In several of the pairplot graphs, different species cluster closely together by features in distinct clusters. This is an easy to predict pattern.
# 2. Continuous data or data where there is a weaker correlation between classification and features. May scale badly with large or high dimensional datasets as all entries need to be checked
# noisy / unclean data with points not included or or outliers skewing results. Non uniform scales.

# TODO: Explore logistic regression, decision trees, SVMs, and non-uniform / non-comparable scales in features (normalising scales)