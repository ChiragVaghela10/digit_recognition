# digit_recognition
This is a hand-written digit recognition project for beginners using K - Nearest Neighbors algorithm. The project also 
evaluate accuracy versus K number of neighbors. The digit picture dataset contains data of 2000 grey scale pictures with
200 images of each 10 digits (0-9). The size of each picture is 16X15 and stored as a string of 240 decimal digits
(ranging 0-6).
<br/>This algorithm is implemented as follows:
1. Image Dataset is reshaped into 10X200x240 for easier indexing. Then it is divided into training data (80%) and 
   testing data (20%).
2. Find the K Nearest Neighbors based on Euclidean Distance between pictures. The value of K = 4 for a single try.
3. The counts of samples of each class is determined, and the sample class with the highest count is assigned to the
   test sample.
4. The class of each test sample is determined similarly.
5. Then accuracy is calculated by the ratio of test samples predicted correctly by the model over total test samples.

Finally, the same algorithm is repeatedly called with various values of K to get a graph between K vs Accuracy.

The following figure shows 10 pictures of each digit (0-9) using data from digit picture dataset file (mfeat-pix.txt):
![Digits Pictures](blob/Digit_Pictures.png)

The following figure shows graph of accuracy vs K neighbors:
![Accuracy vs K](blob/Accuracy_vs_K.png)

### Installation
1. Create a virtual environment.
2. Use `requirements.txt` to install the required libraries as follows:
```
pip install -r requirements.txt
```

### Future Prospects

This project is for beginners to study machine learning algorithms. The digit recognition can be performed by 
implementing other machine learning algorithms like Neural Network, Bayesian Model for understanding and analysis of 
them.
