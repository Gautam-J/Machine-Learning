import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import normalize


def preprocess():
        
    df = pd.read_csv("F:\Gautam\Tech Stuff\Python Projects\Datasets\iris.csv",
                     index_col = False, header = None)

    df.columns = ['Sepal Length', 'Sepal Width', 'Petal Length',
                  'Petal Width' , 'Class']

    df['Class'] = df['Class'].map({'Iris-setosa': 0,
                                   'Iris-versicolor': 1,
                                   'Iris-virginica': 2})

    x = df[df.columns[0:4]].values
    y = df[df.columns[4]].values

    return x, y


x, y = preprocess()

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.2)

train_x = normalize(train_x, norm='l1')
test_x = normalize(test_x, norm='l1')

classifier = KNeighborsClassifier(n_neighbors = 24, p=2, weights='uniform')
classifier.fit(train_x, train_y)
result = classifier.predict(test_x)

for i in result:
    print("Predicted: ", i)

print("\nAccuracy: %0.2f%%" % (classifier.score(test_x, test_y) * 100))

ui = input("Press enter to exit.")
