# digit_recognition
This is a hand-written digit recognition project using K - Nearest Neighbors algorithm. The project also evaluate 
accuracy versus K number of neighbors. The digit picture dataset contains data of 2000 grey scale pictures with 200 
images of each 10 digits (0-9). The size of each picture is 16X15 and stored as a string of 240 decimal digits.
<br/>This algorithm is implemented as follows:
1. Image Dataset is divided into training data (80%) and testing data (20%).
2. Find the K Nearest Neighbors based on Euclidean Distance between pictures. The value of K = 4 for a single try.
3. The counts of samples of each class is determined, and the sample class with the highest count is assigned to the
   test sample.
4. The class of each test sample is determined similarly.
5. Then accuracy is calculated by number of test samples predicted correctly by the model.

Finally, the same algorithm is repeatedly called with various values of K to get a graph between K vs Accuracy.
