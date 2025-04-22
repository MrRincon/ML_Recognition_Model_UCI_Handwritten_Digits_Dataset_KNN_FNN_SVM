//M00774667
package Coursework.algorithms;

//Import any essential packages
import Coursework.validation.Confusion_Matrix;

public class K_Nearest_Neighbour {
	private double kNeighbours;
	private double[][] trainingData;
	private double[][] trainingLabels;

	// Constructor to initialise the KNN algorithm
	public K_Nearest_Neighbour(double[] hyperparameters) {
		this.kNeighbours = hyperparameters[0];
	}

	// Method to train the KNN
	public void train(double[][] trainingData, double[][] trainingLabels) {
		this.trainingData = trainingData;
		this.trainingLabels = trainingLabels;
	}

	// Method to test the KNN and return accuracy
	public double test(double[][] testData, double[][] testLabels, Confusion_Matrix confusionMatrix) {
		int correctPredictions = 0;
		// Looping through each instance in the test data
        for (int testDataIndex = 0; testDataIndex < testData.length; testDataIndex++) {
        	int predictedLabel = predict(testData[testDataIndex]);
        	int actualLabel = getLabelFromOneHot(testLabels[testDataIndex]);
        	// Updating the confusion matrix
        	confusionMatrix.updateMatrix(actualLabel, predictedLabel);
        	// Incrementing the correct predictions if the prediction matches the actual label
        	if(predictedLabel == actualLabel) {
				correctPredictions++;
			}
        }

		return (double) correctPredictions / testData.length * 100.0;
	}

	// Method to predict the label of a testing instance
	private int predict(double[] testingInstance) {
		double[] distances = new double[this.trainingData.length];
		// Calculate distances from the testing instance to all training instances
		for(int trainingDataIndex = 0; trainingDataIndex < this.trainingData.length; trainingDataIndex++) {
			distances[trainingDataIndex] = euclideanDistance(testingInstance, this.trainingData[trainingDataIndex]);
		}
		// Sort indices of distances in ascending order
		int[] neighbourIndices = sortIndicesByValues(distances);
		// Get labels of the nearest k neighbours
		int[] neighbourLabels = new int[(int) this.kNeighbours];
		for(int neighbourIndex = 0; neighbourIndex < this.kNeighbours; neighbourIndex++) {
			neighbourLabels[neighbourIndex] = getLabelFromOneHot(this.trainingLabels[neighbourIndices[neighbourIndex]]);
		}
		return findMostCommonLabel(neighbourLabels);
	}

	// Method to find the most common label among the provided labels
	private int findMostCommonLabel(int[] labels) {
		int maxCount = 0;
		int mostCommonLabel = labels[0];

		// Looping through each label in the array
		for (int label : labels) {
			int count = 0;
			// Comparing the current label with all other labels
			for (int label2 : labels) {
				if (label2 == label) {
                    count++;
                }
			}
			// Updating the most common label if the current label occurs more frequently
			if (count > maxCount) {
				maxCount = count;
				mostCommonLabel = label;
			}
		}
		return mostCommonLabel;
	}

	// Method to calculate the squared Euclidean distance between two feature vectors
	private double euclideanDistance(double[] firstFeature, double[] secondFeature) {
		double sumSquaredDifferences = 0.0;
		for (int valueIndex = 0; valueIndex < firstFeature.length; valueIndex++) {
			sumSquaredDifferences += Math.pow(firstFeature[valueIndex] - secondFeature[valueIndex], 2);
		}

		return Math.sqrt(sumSquaredDifferences);// Return the euclidean distance
	}

	// Method to sort indices based on the corresponding values in the array
	private int[] sortIndicesByValues(double[] values) {
		int[] indices = new int[values.length];
		for (int valueIndex = 0; valueIndex< values.length; valueIndex++) {
			indices[valueIndex] = valueIndex;
		}
		// Performing bubble sort on indices based on the values array
		for(int outerIndex = 0; outerIndex < values.length - 1; outerIndex++) {
			for(int innerIndex = 0; innerIndex < values.length - outerIndex - 1; innerIndex++) {
				if (values[indices[innerIndex]] > values[indices[innerIndex + 1]]) {
					// Swapping indices
					int temp = indices[innerIndex];
					indices[innerIndex] = indices[innerIndex + 1];
					indices[innerIndex + 1] = temp;
				}
			}
		}
		return indices;// Returning sorted indices
	}

	// Method to extract the label from the one-hot encoded array
	private int getLabelFromOneHot(double[] oneHotArray) {
		for (int labelIndex = 0; labelIndex < oneHotArray.length; labelIndex++) {
			if (oneHotArray[labelIndex] == 1.0) {
				return labelIndex;
			}
		}

	 	return -1;// Returns -1 if no active label is found
	}
}