from decorators.decorators import timer
from . import utilities


@timer
def kNN(testSamples, trainingSamples, k=1):
    testedSamples = []
    i = 0
    for testDigits in testSamples:
        j = 0
        for testSample in testDigits:
            k_neighbours = utilities.findNeighbours(testSample, trainingSamples, k)
            # print(k_neighbours)
            testSampleClass = utilities.findSampleClass(k_neighbours)
            testedSamples.append((i, j, testSampleClass))
            j += 1
        i += 1
    # print(testedSamples)
    return utilities.findAccuracy(testedSamples)
