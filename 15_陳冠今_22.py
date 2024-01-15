import pandas as pd
from sklearn import tree
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn import neighbors

iris = datasets.load_iris()

X = pd.DataFrame(iris.data, columns=iris.feature_names)
target = pd.DataFrame(iris.target, columns=["target"])
y = target["target"]

k = 50
knn = neighbors.KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

print(f'KNN model score: {knn.score(X, y)*100}%')

dtree = tree.DecisionTreeClassifier(max_depth = 4)
dtree.fit(X, y)

print(f'Tree model score: {round(dtree.score(X, y)*100,2)}%')

XTrain, XTest, yTrain, yTest = tts(X, y, test_size = 0.4, random_state = 50)

dicti = {}

k = 1
while True:
    try:
        knn = neighbors.KNeighborsClassifier(n_neighbors=k)
        knn.fit(XTrain, yTrain)
        dicti[k] = knn.score(XTest, yTest)
        k += 1
    except:
        break
        
print(f'best rate is {round(max(list(dicti.values()))*100,2)}% when K value is {max(dicti, key = dicti.get)}')

