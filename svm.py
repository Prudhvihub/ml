from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

iris = datasets.load_iris()
#split the data into test and train
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.10, random_state=0)

#create the model and fit the data
model = SVC(kernel='linear', C=1.0)
model.fit(X_train, y_train)

#model testing is predicting the values for test dataset.
y_predicted = model.predict(X_test)
print(classification_report(y_test, y_predicted))