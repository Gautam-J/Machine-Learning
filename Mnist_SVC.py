import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm
import random

digits = datasets.load_digits()
img, x, y = digits.images[-100:], digits.data, digits.target

train_x, train_y = digits.data[:-100], digits.target[:-100]
test_x, test_y = digits.data[-100:], digits.target[-100:]

clf = svm.SVC(gamma=0.0001, C=100)
clf.fit(train_x, train_y)

score = clf.score(test_x, test_y)
print('Accuracy: %0.2f%%' % (score * 100))

while True:
    num = random.randint(0, 100)
    pred = clf.predict([test_x[num]])

    plt.imshow(img[num], cmap=plt.cm.gray_r, interpolation='nearest')
    plt.title('Predicted: {} Actual: {}'.format(pred, test_y[num]))
    plt.show()
