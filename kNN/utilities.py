import operator
import collections as cl


def euDistance(pointA, pointB):
    distance = pow((pointA - pointB), 2)
    return pow(sum(distance), 0.5)


def findNeighbours(testSample, trainData, k):
    distances = []

    i = 0
    for trainDigit in trainData:
        j = 0
        for trainSample in trainDigit:
            distance = euDistance(testSample, trainSample)
            distances.append((i, j, distance))
            j += 1
        i += 1
    # Sort distances based on distance, which is on index 2
    sorted_list = sorted(distances, key=operator.itemgetter(2))
    # print(len(sorted_list))
    return sorted_list[:k]


def findSampleClass(neighbours):
    # Counting is done based on row (0th index) of tuple
    classes_with_freq = cl.Counter([x[0] for x in neighbours])
    # print(classes_with_freq)

    # Key (class of sample) is obtained based on highest value (count of neighbours)
    # NOTE: If 2 classes are having same count than first class will be chosen as sample class
    sampleClass = max(classes_with_freq.items(), key=operator.itemgetter(1))[0]
    return sampleClass


def findAccuracy(testedSamples):
    correctPredictions = 0
    for prediction in testedSamples:
        if prediction[0] == prediction[2]:
            correctPredictions += 1
    return correctPredictions / len(testedSamples)
