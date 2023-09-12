from sklearn import linear_model
import numpy as np


#create the training data set, with 2 input integers
input_data = np.random.randint(50, size=(20,2))
input_sum = np.zeros(len(input_data))
for row in range(len(input_data)):
    input_sum[row] = input_data[row][0] + input_data[row][1]

#build a linear regression model with the training data
linear_regression = linear_model.LinearRegression(fit_intercept=False)
linear_regression.fit(input_data, input_sum)

#on test data 
predicted_sum = linear_regression.predict([[991231, 30]])
print('the predicted sum is: ' + str(predicted_sum))