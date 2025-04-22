//M00774667
package Coursework.validation;

public class Confusion_Matrix {
	private int[][] matrix;// 2D array for the confusion matrix
	private int numLabels;// number of labels

	// Constructor for the confusion matrix based on the number of labels
	public Confusion_Matrix(int numLabels) {
		this.matrix = new int[numLabels][numLabels];
		this.numLabels = numLabels;
	}

	// Method to update the confusion matrix with an actual and predicted labels
	public void updateMatrix(int actualLabel, int predictedLabel) {
		this.matrix[actualLabel][predictedLabel]++;// Increments the count for the given actual-predicted pair
	}

	// Method to calculate precision for each label
	public double[] calculatePrecision() {
		double[] precision = new double[this.numLabels];
		// Looping over each actual label to calculate precision
		for (int actualLabel = 0; actualLabel < this.numLabels; actualLabel++) {
			int truePositive = this.matrix[actualLabel][actualLabel];// Correcting the count of true positives for the label
			int falsePositive = 0;
			// Sum of the false positives labels predicted of the current actual label
			for (int predictedLabel = 0; predictedLabel < this.numLabels; predictedLabel++) {
				if (actualLabel != predictedLabel) {
					falsePositive += this.matrix[predictedLabel][actualLabel];// Incrementing the false positives
				}
			}
			// Calculating the precision and handling division by zero
			if (truePositive + falsePositive == 0) {
				precision[actualLabel] = 0;// Avoid division by zero
			} else {
				precision[actualLabel] = (double) truePositive / (truePositive + falsePositive);
			}

		}

		return precision;// Returns the precision values for all labels
	}

	// Method to calculate recall for each label
	public double[] calculateRecall() {
		double[] recall = new double[this.numLabels];
		// Looping over each actual label to calculate recall
		for (int actualLabel = 0; actualLabel < this.numLabels; actualLabel++) {
			int truePositive = matrix[actualLabel][actualLabel];// Correcting the count of true positives for the label
			int falseNegative = 0;
			// Sum of the false negatives labels predicted for the current actual label
			for (int predictedLabel = 0; predictedLabel < this.numLabels; predictedLabel++) {
				if (actualLabel != predictedLabel) {
					falseNegative += matrix[actualLabel][predictedLabel];
				}
			}
			// Calculating the recall and handling division by zero
			if (truePositive + falseNegative == 0) {
				recall[actualLabel] = 0;// Avoid division by zero
			} else {
				recall[actualLabel] = (double) truePositive / (truePositive + falseNegative);
			}
		}

		return recall;// Returns the recall values for all labels
	}

	// Method to calculate F1-score for each label
	public double[] calculateF1Score() {
		double[] f1Scores = new double[this.numLabels];
		double[] precision = calculatePrecision();
		double[] recall = calculateRecall();
		// Looping over each actual label to calculate F1-score
		for (int actualLabel = 0; actualLabel < this.numLabels; actualLabel++) {
			// Calculating F1-score using precision and recall, and handling division by zero
			if (precision[actualLabel] + recall[actualLabel] == 0) {
				f1Scores[actualLabel] = 0;
			} else {
				f1Scores[actualLabel] = 2 * (precision[actualLabel] * recall[actualLabel])
						/ (precision[actualLabel] + recall[actualLabel]);
			}
		}

		return f1Scores;// Returns the F1-score values for all labels
	}

	// Method to display the confusion matrix and calculation of metrics in a formatted string
	@Override
	public String toString() {
		int boxWidth = 115;
		StringBuilder sb = new StringBuilder();
		String title = "Confusion Matrix";
		int padding = (boxWidth - 2 - title.length()) / 2;
		sb.append("|").append(" ".repeat(padding)).append(title).append(" ".repeat(padding));
		if ((boxWidth - 2 - title.length()) % 2 != 0) {
			sb.append(" ");
		}
		sb.append("|\n").append("=".repeat(boxWidth)).append("\n");
		// Column headers with predicted labels
		sb.append(String.format("| %-12s", ""));
		for (int label = 0; label < this.numLabels; label++) {
			sb.append(String.format("| %-7s ", "Pred " + label));
		}
		sb.append("|\n").append("-".repeat(boxWidth)).append("\n");
		// Rows with actual labels and their corresponding predictions
		for (int actualLabel = 0; actualLabel < this.numLabels; actualLabel++) {
			sb.append(String.format("|  Actual  %d  |", actualLabel));
			for (int predictedLabel = 0; predictedLabel < this.numLabels; predictedLabel++) {
				sb.append(String.format(" %7d |", this.matrix[actualLabel][predictedLabel]));
			}
			sb.append("\n");
		}
		sb.append("=".repeat(boxWidth)).append("\n");
		// Calculating and displaying Precision, Recall and F1-score for each label
		double[] precision = calculatePrecision();
		double[] recall = calculateRecall();
		double[] f1Score = calculateF1Score();
		sb.append(String.format("| %-18s | %-28s | %-28s | %-28s |\n", "Label", "Precision", "Recall",
				"F1-Score"));
		sb.append("=".repeat(boxWidth)).append("\n");
		for (int label = 0; label < this.numLabels; label++) {
			sb.append(String.format("| %-18s | %27.4f%% | %27.4f%% | %27.4f%% |\n",
					"Label " + label, precision[label] * 100, recall[label] * 100, f1Score[label] * 100));
		}
		sb.append("=".repeat(boxWidth)).append("\n");

		return sb.toString();// Returns the formatted string
	}

}