# Hand Written Digit Recognition

This is a hand written digit recognition project for machine learning enthusiasts to get hands-on experience of the 
basic machine learning algorithms. The project demonstrates internal mechanism of the common machine learning
algorithms. The implementation intentionally does not include sophisticated ML libraries to achieve this objective.

## Project Scope

The digit picture dataset contains data of 2000 grey scale pictures with 200 images of each 10 digits (0-9). The size of
each picture is 16X15 and stored as a string of 240 decimal digits (ranging 0-6).

The following figure shows 10 pictures of each digit (0-9) using data from digit picture dataset file (mfeat-pix.txt):
<br/><br/>
![Digits Pictures](blob/Digit_Pictures.png)

### Data Preprocessing

- Image Dataset is initially reshaped into 10X200x240 to collect all digits' samples into training and test data split.
- Split ratio is 0.8 for all algorithms
- Each row has 240 values in tables below. Table A depicting **original dataset** is (2000, 240) & _reshaped_ into Table B 
with dimensions (10, 200, 240).
- Each row represents one image of a digit of size 16 X 15 = 240.

**TABLE A:**

| index | image_data                  |
|-------|-----------------------------|
| 0     | <------ for digit 0 ------> |
| 1     | <------     ""      ------> |
| .     | <------     ""      ------> |
| 199   | <------ for digit 0 ------> |
| 200   | <------ for digit 1 ------> |
| 201   | <------     ""      ------> |
| ...   | <------     ""      ------> |
| 1999  | <------ for digit 9 ------> |

            |
           \|/

**TABLE B:**

|   | index | image_data              |
|---|-------|-------------------------|
| 0 | 0     | <------ digit 0 ------> |
| 0 | 1     | <------   ""   -------> |
| . | .     | <------   ""    ------> |
| 0 | 199   | <------ digit 0 ------> |
| 1 | 0     | <------ digit 1 ------> |
| 1 | 1     | <------   ""    ------> |
| . | .     | <------   ""    ------> |
| 1 | 199   | <------ digit 1 ------> |
| . | .     | <------   ""    ------> |
| 9 | 0     | <------ digit 9 ------> |
| . | .     | <------   ""    ------> |
| 9 | 199   | <------ digit 9 ------> |

- Then TABLE B is split according to split ratio to obtain (10 X 160, 240) and (10 X 40 X 240) containing samples of
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

The model can be considered as neural network of 1 layer having 10 nodes and 'linear' activation function.

### Results

<img src="results/lr_cost.png" alt="random_sample" width="240" height="240"/>
<img src="results/lr_result.png" alt="random_sample" width="240" height="240"/>
<img src="results/lr_random_sample.png" alt="random_sample" width="240" height="240"/>

## K-Nearest Neighbours

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
![Accuracy vs K](blob/Accuracy_vs_K.png)

##### Findings

The plot is an illustration of 'Elbow Method' and shows that the accuracy peaks when k is in range of 5 - 10. Then the
accuracy sharply decreases indicating accuracy peak as overfit of the model. The sharp decrease in accuracy is due
to the generalization of model. The further slow decrease in accuracy is supported by the fact that overall cost
function is non-increasing function which is to calculate minimum distance between mean (centroid) and the data points. 
The elbow curve is **not a smooth decreasing function** indicating reliability of the chosen method to determine optimum
value of K. The reason for non-smoothness and more like elbow shape is because each handwritten digit group (group of 
samples indicating 0s, 1s and so on) has distinct features of that particular digit present in the data (i.e. numerical
representation) and when K reaches near the optimum value, the model generalizes. This is a trade-off for the decrease
in train cost vs increase in test cost leading to little bit lesser accuracy than smaller values of K (overfit case).

## Installation
1. Create a virtual environment.
2. Use `requirements.txt` to install the required libraries as follows:
```
pip install -r requirements.txt
```

## Future Prospects
...
