//M00774667
package Coursework.algorithms;

public class Hyperparameters {
	private String algorithm;
	private double[][] hyperparameterCombinations;
	// Hyperparameters for a Support Vector Machine (SVM)
	private double[] gammaValues;
	private double[] cValues;
	private int maxIterations;
	private double tolerance;
	// Hyperparameters for a Feed-forward Neural Network (FNN)
	private int[] hiddenNeurons;
	private int[] epochs;
	private double[] learningRates;
	// Hyperparameters for a K-Nearest Neighbour (KNN)
	private int[] kValues;

	// Constructor for hyperparameter values for each machine learning algorithm
	public Hyperparameters(String algorithm) {
		this.algorithm = algorithm;
		if (this.algorithm == "SVM") {
			// Best possible values for Gamma, C, Maximum Iterations and Stopping criterion for the SVM
			this.gammaValues = new double[] { 1, 10, 100 };
			this.cValues = new double[] { 1, 10, 100 };
			this.maxIterations = 10;
			this.tolerance = 1e-5;
			int totalCombinations = this.gammaValues.length * this.cValues.length;
			this.hyperparameterCombinations = new double[totalCombinations][4];
		} else if (this.algorithm == "FNN") {
			// Best possible values for Hidden neurons, Epochs and Learning rate for the FNN
			this.hiddenNeurons = new int[] { 250 };
			this.epochs = new int[] { 750 };
			this.learningRates = new double[] { 0.01, 0.05 };
			int totalCombinations = this.hiddenNeurons.length * this.learningRates.length * this.epochs.length;
			this.hyperparameterCombinations = new double[totalCombinations][3];
		} else if (this.algorithm == "KNN") {
			// Possible values for K for the KNN
			this.kValues = new int[] {1,2,3,4,5,6,7,8,9,10};
			int totalCombinations = this.kValues.length;
			this.hyperparameterCombinations = new double[totalCombinations][1];
		}
	}

	// Method to get the hyperparameters combinations
	public double[][] getHyperparameterCombinations() {
		int combinationIndex = 0;
		if(this.algorithm == "SVM") {
			// Storing the combinations of Gamma, C, Maximum Iterations and Tolerance values
			for (double gammaValue : this.gammaValues) {
				for (double cValue : this.cValues) {
					this.hyperparameterCombinations[combinationIndex][0] = gammaValue;
					this.hyperparameterCombinations[combinationIndex][1] = cValue;
					this.hyperparameterCombinations[combinationIndex][2] = this.maxIterations;
					this.hyperparameterCombinations[combinationIndex][3] = this.tolerance;
					combinationIndex++;
				}
			}
		} else if (this.algorithm == "FNN") {
			// Storing the combinations of Hidden Neurons, Epochs and Learning Rates values
			for (int hiddenNeuron : this.hiddenNeurons) {
				for (double learningRate : this.learningRates) {
					for (int epoch : epochs) {
						this.hyperparameterCombinations[combinationIndex][0] = hiddenNeuron;
						this.hyperparameterCombinations[combinationIndex][1] = learningRate;
						this.hyperparameterCombinations[combinationIndex][2] = epoch;
						combinationIndex++;
					}
				}
			}
		} else if (this.algorithm == "KNN") {
			// Storing the number of possible Nearest Neighbour values
			for (int kValue : this.kValues) {
				this.hyperparameterCombinations[combinationIndex][0] = kValue;
				combinationIndex++;
			}
		}
		return this.hyperparameterCombinations;
	}
}