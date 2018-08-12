import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

data = load_boston()
x, y, = data.data, data.target

train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

clf = LinearRegression()
clf.fit(train_x, train_y)

score = clf.score(test_x, test_y)
pred = clf.predict(test_x)
mse = mean_squared_error(test_y, pred)

print('Accuracy %0.2f%%' % (score * 100))
print('Error: ', mse)

plt.scatter(pred, test_y)
plt.xlabel('Actual Price')
plt.ylabel('Prediction')
plt.title('Boston Housing Price ($1000s)')
plt.show()
