//M00774667
package Coursework.validation;

import Coursework.algorithms.Feedforward_Neural_Network;
import Coursework.algorithms.K_Nearest_Neighbour;
//Import any essential packages
import Coursework.algorithms.Support_Vector_Machine;

public class K_Fold_Cross_Validation {
	private String algorithm;
	private int kValues;// Number of folds for the cross-validation
	private double[] hyperparameters;// Hyperparameters
	private double averageAccuracy;// Average accuracy after cross-validation
	private double[][] dataFile;// 2D array for the input data
	private double[][] allOneHotLabels; // Array for the one hot encoded labels
	private Confusion_Matrix confusionMatrix;// Confusion matrix to evaluate model performance

	// Constructor for the K-Fold Cross-Validation with provided hyperparameters
	public K_Fold_Cross_Validation(double[][] dataFile, double[][] allOneHotLabels, double[] hyperparameters,
			String algorithm) {
		this.algorithm = algorithm;
		this.kValues = 2;// Default number of folds set to 2
		this.dataFile = dataFile;
		this.allOneHotLabels = allOneHotLabels;
		this.averageAccuracy = 0.0;
		this.hyperparameters = hyperparameters;
		this.confusionMatrix = new Confusion_Matrix(10);// Initialisation of Confusion Matrix for a 10-class classification
	}

	// Method to run the K-Fold Cross-Validation process
	public double kFoldRun() {
		// Split data and output labels/neurons into K amount of folds
		double[][][] dataFolds = getDataFolds(this.dataFile);
		double[][][] labelFolds = getDataFolds(this.allOneHotLabels);
		double totalAccuracy = 0.0;
		// Looping over each fold as the test set
		for (int currentFold = 0; currentFold < this.kValues; currentFold++) {
			// Using all the other folds except the current one as the training set
			double[][] trainingData = allFoldsExceptOne(dataFolds, currentFold);
			double[][] trainingLabels = allFoldsExceptOne(labelFolds, currentFold);
			// Using the current fold as the testing set
			double[][] testingData = dataFolds[currentFold];
			double[][] testingLabels = labelFolds[currentFold];
			// Initialising each machine learning system with their corresponding parameters to evaluate them all
			if (algorithm == "FNN") {
				Feedforward_Neural_Network fnn = new Feedforward_Neural_Network(this.hyperparameters);
				fnn.train(trainingData, trainingLabels);
				double accuracy = fnn.test(testingData, testingLabels, this.confusionMatrix);
				totalAccuracy += accuracy;
			} else if (algorithm == "SVM") {
				Support_Vector_Machine svm = new Support_Vector_Machine(this.hyperparameters);
				svm.train(trainingData, trainingLabels);
				double accuracy = svm.test(testingData, testingLabels, this.confusionMatrix);
				totalAccuracy += accuracy;
			} else if (algorithm == "KNN") {
				K_Nearest_Neighbour knn = new K_Nearest_Neighbour(this.hyperparameters);
				knn.train(trainingData, trainingLabels);
				double accuracy = knn.test(testingData, testingLabels, this.confusionMatrix);
				totalAccuracy += accuracy;
			}
		}
		this.averageAccuracy = totalAccuracy / this.kValues;// Calculation and storage of average accuracy across all folds
		// System.out.println(toString());// Printing the cross-validation results summary (to monitor the cross-validation)
		return this.averageAccuracy;// Returns the average accuracy
	}

	// Method to split the data in k equal folds
	private double[][][] getDataFolds(double[][] data) {
		int foldSize = data.length / this.kValues;
		double[][][] dataFolds = new double[this.kValues][][];
		// Looping over each fold to create subsets of data
		for (int foldIndex = 0; foldIndex < this.kValues; foldIndex++) {
			int startIndex = foldIndex * foldSize;
			int endIndex;
			// Including all the remaining data in the last fold
			if (foldIndex == this.kValues - 1) {
				endIndex = data.length;
			} else {
				endIndex = (foldIndex + 1) * foldSize;
			}
			int foldLength = endIndex - startIndex;
			dataFolds[foldIndex] = new double[foldLength][];
			System.arraycopy(data, startIndex, dataFolds[foldIndex], 0, foldLength);
		}
		return dataFolds;// Returns the split data in folds
	}

	// Method to combine all the folds except the one to be tested
	private double[][] allFoldsExceptOne(double[][][] allFolds, int foldToExclude) {
		int totalRows = 0;
		// Calculating the total number of rows excluding the specified fold
		for (int foldIndex = 0; foldIndex < allFolds.length; foldIndex++) {
			if (foldIndex != foldToExclude) {
				totalRows += allFolds[foldIndex].length;
			}
		}
		double[][] combinedFolds = new double[totalRows][];// Holding the combined folds in an array
		int currentRowIndex = 0;
		// Looping to copy all the data folds except the specified one
		for (int foldIndex = 0; foldIndex < allFolds.length; foldIndex++) {
			if (foldIndex != foldToExclude) {
				System.arraycopy(allFolds[foldIndex], 0, combinedFolds, currentRowIndex, allFolds[foldIndex].length);
				currentRowIndex += allFolds[foldIndex].length;
			}
		}

		return combinedFolds;// Returns the combined data
	}

	// Method to get the confusion matrix
	public void getConfusionMatrix() {
		System.out.println(this.confusionMatrix.toString());
	}

	// Method to display the summary of the cross-validation and the results in a formatted string
	@Override
	public String toString() {
		int boxWidth = 115;
		String title = "";
		int padding;
		StringBuilder sb = new StringBuilder();
		sb.append("=".repeat(boxWidth)).append("\n");
		if (this.algorithm == "SVM") {
			title = String.format("MACHINE LEARNING --- %s (%s)", "Support Vector Machine",algorithm);
		} else if (this.algorithm == "FNN"){
			title = String.format("MACHINE LEARNING --- %s (%s)", "Feedforward Neural Network",algorithm);
		} else if (this.algorithm == "KNN") {
			title = String.format("MACHINE LEARNING --- %s (%s)", "K-Nearest Neighbour",algorithm);
		}
		padding = (boxWidth - 2 - title.length()) / 2;
		sb.append("|").append(" ".repeat(padding)).append(title).append(" ".repeat(padding));
		if ((boxWidth - 2 - title.length()) % 2 != 0) {
			sb.append(" ");
		}
		sb.append("|\n");
		sb.append("=".repeat(boxWidth)).append("\n");
		title = String.format("K-Fold Cross-Validation --- %d - Fold", this.kValues);
		padding = (boxWidth - 2 - title.length()) / 2;
		sb.append("|").append(" ".repeat(padding)).append(title).append(" ".repeat(padding));
		if ((boxWidth - 2 - title.length()) % 2 != 0) {
			sb.append(" ");
		}
		sb.append("|\n");
		sb.append("=".repeat(boxWidth)).append("\n");
		sb.append(String.format("| %-111s |\n", "Parameters"));
		if (this.algorithm == "SVM") {
			sb.append(
					String.format("| %-111s |\n", String.format("Values = 64 | Labels = 10 | Gamma = %.3f | C = %.3f | Maximum Iterations = %.2f | Tolerance = %.5f",
							this.hyperparameters[0], this.hyperparameters[1], this.hyperparameters[2], this.hyperparameters[3])));
		} else if (this.algorithm == "FNN") {
			sb.append(String.format("| %-111s |\n", String.format(
					"Input Neurons = 64 | Output Neurons = 10 | Hidden Neurons = %.2f | Learning Rate = %.3f | Epochs = %.2f",
					this.hyperparameters[0], this.hyperparameters[1], this.hyperparameters[2])));
		} else if (this.algorithm == "KNN") {
			sb.append(String.format("| %-111s |\n", String.format(
					"Values = 64 | Labels = 10 | K value = %.2f",
					this.hyperparameters[0])));
		}
		sb.append(String.format("| %-111s |\n", String.format("Total accuracy = %.4f%%", this.averageAccuracy)));
		sb.append("=".repeat(boxWidth));

		return sb.toString();// Returns the formatted string
	}

}