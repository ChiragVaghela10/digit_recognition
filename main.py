import pandas as pd
import matplotlib.pyplot as plt

from kNN import algorithms
from contants import *
with open('data/mfeat-pix.txt', 'r', encoding='ascii') as dataFile:
    mfeat_pix = pd.read_table(dataFile, sep='  ', header=None, engine='python').values


# Image Dataset is reshaped into 10X200x240 for easier indexing.
new_shape = mfeat_pix.reshape(total_digits, digits_repetition, image_size)


def plot(columns):
    """
    function to draw pictures from the data vectors
    """
    for i in range(columns):
        for j in range(columns):
            pic = mfeat_pix[digits_repetition * i + j][:]
            picmat_reverse = -pic
            picmat = picmat_reverse.reshape(16, 15)
            plt.figure(1, figsize=(11, 6.5))
            # TODO: Come up with dynamic formula
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.axis('off')
            plt.imshow(picmat, cmap='gray')
    plt.show()


def accuracies(trainData, testData, k_values):
    """
    function get list of k values as parameter and return corresponding accuracy values list

    trainData: training data, first 160 images of each digit
    testData: testing data, remaining 40 images of each digit
    k_list: list of values of K for which the accuracies of KNN is to found
    return: accuracy_list corresponding to list of values of K neighbors, max_result value for its corresponding K
    neighbors value
    """
    result_list = []
    max_result = 0
    associated_k = 0
    for k in k_values:
        accuracy = algorithms.kNN(testData, trainData, k)
        result_list.append(accuracy)
        if accuracy > max_result:
            max_result = accuracy
            associated_k = k
    return result_list, max_result, associated_k


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # training data: first 160 images of each digit
    train_data = new_shape[:, :int(digits_repetition * trainRatio), :]
    # testing data: remaining 40 images of each digit
    test_data = new_shape[:, int(digits_repetition * trainRatio):, :]
    # Shows 10 pictures of each digit
    plot(10)

    # (a) Single execution with K = 3
    print('Running KNN model...')
    print('Accuracy of the kNN model (for K = 4) is: ', algorithms.kNN(test_data, train_data, 3))

    # TODO: Implement 10-fold cross validation
    # (b) 10-fold Cross Validation
    pass

    # (c) Run kNN: k vs accuracy
    k_list = [2, 3, 4, 6, 9, 11, 12, 13, 14, 15, 16, 17, 18, 20, 25, 30, 50, 70, 90, 100]
    print('Running KNN Model for each value of K. So, sit back and relax...')
    accuracy_list, max_accuracy, max_k = accuracies(train_data, test_data, k_list)
    k_accuracy_list = zip(k_list, accuracy_list)
    print('K vs accuracy values are:', list(k_accuracy_list))
    print('Max accuracy is:', max_accuracy, 'when k =', max_k)

    plt.plot(k_list, accuracy_list)
    plt.title('Accuracy vs k')
    plt.xlabel('k Neighbors')
    plt.ylabel('Accuracy')
    plt.show()
