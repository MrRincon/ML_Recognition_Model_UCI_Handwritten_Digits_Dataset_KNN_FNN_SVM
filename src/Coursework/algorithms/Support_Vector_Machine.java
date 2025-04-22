//M00774667
package Coursework.algorithms;

//Import any essential packages
import Coursework.validation.Confusion_Matrix;

public class Support_Vector_Machine {
    private double[] hyperparameters;// Array of hyperparameters
    private Binary_SVM[] classifiers;// Array of binary SVM classifiers use d in the one-vs-one classification
    private int[][] labelPairs;// One-vs-One label pairs

    //Constructor for the support vector machine with the necessary parameters
    public Support_Vector_Machine(double[] hyperparameters) {
    	this.hyperparameters = hyperparameters;
    }

    // Method to train the SVM using one-vs-one approach
    public void train(double[][] trainingData, double[][] trainingLabels) {
        int numLabels = trainingLabels[0].length;// Number of unique labels in the data
        int totalClassifiers = (numLabels * (numLabels - 1)) / 2;// Total number of classifiers needed for one-vs-one
        classifiers = new Binary_SVM[totalClassifiers];// Classifiers array with a length of the classifiers needed
        labelPairs = new int[totalClassifiers][2];// Array for the label pairs used in the one-vs-one approach
        int classifierIndex = 0;
        // Looping for one-vs-one classification over all possible label pairs
        for (int labelOne = 0; labelOne < numLabels; labelOne++) {
            for (int labelTwo = labelOne + 1; labelTwo < numLabels; labelTwo++) {
            	// Extracting data for binary classification for the current label pair
                double[][] binaryData = filterBinary(trainingData, trainingLabels, labelOne, labelTwo);
                int[] binaryLabels = filterBinary(trainingLabels, labelOne, labelTwo);
                // Creating and training a binary SVM for the current label pair
                classifiers[classifierIndex] = new Binary_SVM(this.hyperparameters);
                classifiers[classifierIndex].train(binaryData, binaryLabels);
                // Storing the current label pair
                labelPairs[classifierIndex] = new int[] {labelOne, labelTwo};
                classifierIndex++;
            }
        }
    }

    // Method to test the SVM and evaluate its accuracy using a confusion matrix
    public double test(double[][] testData, double[][] testLabels, Confusion_Matrix confusionMatrix) {
        int correctPredictions = 0;
        // Looping through each instance in the test data
        for (int testDataIndex = 0; testDataIndex < testData.length; testDataIndex++) {
            double[] instance = testData[testDataIndex];
            // Extract the true label from the one-hot encoded label array
            int trueLabel = getLabelFromOneHot(testLabels[testDataIndex]);
            // One-vs-One voting for the current instance
            int[] votes = new int[10];// Array to store the votes for each label
            for (int classifierIndex = 0; classifierIndex < classifiers.length; classifierIndex++) {
                int labelOne = labelPairs[classifierIndex][0];
                int labelTwo = labelPairs[classifierIndex][1];
                // Predicting the label using the current binary classifier
                int prediction = classifiers[classifierIndex].predict(instance);
                // Increment the vote count for the predicted label
                if (prediction == 1) {
                    votes[labelOne]++;
                } else {
                    votes[labelTwo]++;
                }
            }
            // Getting the label with the most votes
            int predictedLabel = 0;
            int maxVotes = votes[0];
            for (int voteIndex = 1; voteIndex < votes.length; voteIndex++) {
                if (votes[voteIndex] > maxVotes) {
                    maxVotes = votes[voteIndex];
                    predictedLabel = voteIndex;
                }
            }
            // Updating the confusion matrix with the actual and predicted labels
            confusionMatrix.updateMatrix(trueLabel, predictedLabel);
            // Incrementing the correct predictions if the prediction matches the actual label
            if (trueLabel == predictedLabel) {
				correctPredictions++;
			}
        }
        // Returning the accuracy as a percentage
        return (double) correctPredictions / testData.length * 100.0;
    }

    // Method to filter the training data for two specific labels
    private double[][] filterBinary(double[][] data, double[][] labels, int labelOne, int labelTwo) {
    	int count = 0;
    	// Counting the matching labels
    	for (double[] label : labels) {
    		int Label = getLabelFromOneHot(label);
    		if(Label == labelOne || Label == labelTwo) {
    			count++;
    		}
    	}
    	// Storing the filtered data into a new array
    	double[][] binaryData = new double[count][data[0].length];
    	int index = 0;
    	for (int dataIndex = 0; dataIndex < labels.length; dataIndex++) {
    		int Label = getLabelFromOneHot(labels[dataIndex]);
    		if (Label == labelOne || Label == labelTwo) {
    			binaryData[index++] = data[dataIndex];
    		}
    	}
        return binaryData;
    }

    // Overload Method to filter the training labels for two specific labels
    private int[] filterBinary(double[][] labels, int labelOne, int labelTwo) {
    	int count = 0;
    	// Counting the matching labels
    	for (double[] label : labels) {
    		int Label = getLabelFromOneHot(label);
    		if(Label == labelOne || Label == labelTwo) {
    			count++;
    		}
    	}
    	// Storing the filtered labels into a new array
    	int[] binaryLabels = new int[count];
    	int index = 0;
    	for (double[] label : labels) {
    		int Label = getLabelFromOneHot(label);
    		if(Label == labelOne) {
    			binaryLabels[index++] = 1 ;
    		} else if (Label == labelTwo) {
    			binaryLabels[index++] = -1;
    		}
    	}
        return binaryLabels;
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