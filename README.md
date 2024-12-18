# Hand Written Digit Recognition

This is a hand written digit recognition project for machine learning enthusiasts to get hands-on experience of the 
basic machine learning algorithms. The project demonstrates internal mechanism of the common machine learning
algorithms. The implementation intentionally does not include sophisticated ML libraries to achieve this objective.

## Project Scope

The digit picture dataset contains data of 2000 grey scale pictures with 200 images of each 10 digits (0-9). The size of
each picture is 16X15 and stored as a array of 240 integers (ranging 0-6).

The following figure shows 10 pictures of each digit (0-9) using data from digit picture dataset file (mfeat-pix.txt):
<br/><br/>
![Digits Pictures](old_code/blob/Digit_Pictures.png)

### Data Preprocessing

- Image Dataset is initially reshaped into (10 X 200 x 240) to collect all digits' samples into training and test data split.
- Split ratio is 0.8 for all algorithms
- Each row has 240 values in tables below. Table A depicting **original dataset** is (2000 X 240) & _reshaped_ into Table B 
with dimensions (10 X 200 X 240).
- Each row represents one image of a digit of size 16 X 15 = 240.
- Then TABLE B is split according to split ratio to obtain (10 X 160 X 240) and (10 X 40 X 240) containing samples of
all digits for training and testing.
- Finally, reshaped back to **Train Data** (1600 X 240) and **Test Data** (400 X 240).


## Linear Regression

### Algorithm
Linear Regression is preferred to predict real values. However, we can use it for classification task but with
limitations in performance. Therefore, in this implementation separate linear function for each digit is used to improve 
performance.

This implementation uses 10 linear functions to predict each digit {0, 1, ..., 9}. e.g. First function would have its own 
weights (240,) and bias (1,) to predict digit '0'. Second function would have its own to predict digit '1' and so on...

Maximum of predicted values (10,) obtained by model is used to determine the predicted digit. e.g. when 
prediction of the model is [0.0234, 0.0263, 0.0163, 8.462, 0.1643, 0.2351, 0.0432, 0.2785, 0.5488, 0.0462], the
predicted digit would be '3'.

The model can be considered as neural network of 1 layer having 10 nodes and 'linear' activation function but without
forward and backward propagation.

### Results
Multiple Linear Regressions (10) were trained for 500 iterations to get average prediction accuracy of 99.0 % for all 
digits.

Cost vs Iterations                         |               Accuracy               | Random Sample                                |
:-----------------------------------------:|:------------------------------------:|:---------------------------------------------|
![Cost vs Iterations](results/linear_regression_cost.png) |  ![Accuracy](results/linear_regression_result.png)  | ![Predictions](results/lr_random_sample.png) |


## Logistic Regression

### Algorithm
Multiple Logistic Regressions (10) units were trained for predicting digits. Logistic Units are better suitable to
detect digits. Since, each unit would have binary (Yes/No) output.

The model was trained for 500 iterations to get average prediction accuracy of 99.1 % for all digits.

### Results
Cost vs Iterations                         |               Accuracy               | Curve Approximation vs Iterations                           |
:-----------------------------------------:|:------------------------------------:|:------------------------------------------------------------|
![Cost vs Iterations](results/logistic_regression_cost.png) |  ![Accuracy](results/logistic_regression_result.png)  | ![Random Sample](ml/experiments/plots/lr_learning_plot.png) |

## K-Nearest Neighbours
NOTE: In progress...
### Algorithm

1. Find the K Nearest Neighbors based on Euclidean Distance between pictures. The value of K = 4 for a single try.
2. The counts of samples of each class is determined, and the sample class with the highest count is assigned to the
   test sample.
3. The class of each test sample is determined similarly.
4. Then accuracy is calculated by the ratio of test samples predicted correctly by the model over total test samples.

Finally, the same algorithm is repeatedly called with various values of K to get a graph between K vs Accuracy.

The project also evaluate accuracy versus K number of neighbors.

### Results

The following figure shows graph of accuracy vs K neighbors:<br/><br/>
![Accuracy vs K](old_code/blob/Accuracy_vs_K.png)

## Installation
1. Create a virtual environment.
2. Use `requirements.txt` to install the required libraries as follows:
```
pip install -r requirements.txt
```

## Future Prospects
In progress...
