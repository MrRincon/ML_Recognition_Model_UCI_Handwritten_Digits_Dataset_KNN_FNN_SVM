//M00774667
package Coursework.algorithms;

public class Binary_SVM {
	// SVM hyperparameters
	private double gammaValue;// Parameter for the RBF kernel
	private double cValue;// Regularisation parameter
	private double maxIterations;// Maximum number of iterations for the Sequential Minimal Optimisation (SMO) algorithm
	private double tolerance;// Convergence tolerance for optimisation
	// Model parameters
	private double[] alphas;// Lagrange multipliers for each training instance
	private double b;// Bias term for the decision boundary
	private double[][] trainingData;// Training data
	private int[] trainingLabels;// Training labels

	// Constructor for the SVM with provided hyperparameters
	public Binary_SVM(double[] hyperparameters) {
		this.gammaValue = hyperparameters[0];
		this.cValue = hyperparameters[1];
		this.maxIterations = hyperparameters[2];
		this.tolerance = hyperparameters[3];
	}

	// Method to train the binary SVM using
	public void train(double[][] data, int[] labels) {
		this.trainingData = data;
		this.trainingLabels = labels;
		int n = this.trainingData.length;
		this.alphas = new double[n];
		this.b = 0.0;
		// Implementing simplified SMO to the training process
		for (int iteration = 0; iteration < this.maxIterations; iteration++) {
			for (int currentIndex = 0; currentIndex < n; currentIndex++) {
				// Calculating the error for the current instance
				double errorCurrent = calculateError(this.trainingData[currentIndex], this.trainingLabels[currentIndex]);
				// Checking if the current alpha violates the Karush-Kuhn-Tucker(KKT) conditions after the first iteration
				if (iteration > 0) {
					if ((this.trainingLabels[currentIndex] * errorCurrent < -this.tolerance && alphas[currentIndex] < this.cValue)
							|| (this.trainingLabels[currentIndex] * errorCurrent > this.tolerance && alphas[currentIndex] > 0)) {
						continue;
					}
				}
				// Selecting a second point randomly for optimisation
				int pairIndex = (currentIndex + 1) % n;
				double errorPair = calculateError(this.trainingData[pairIndex], this.trainingLabels[pairIndex]);
				// Saving the old alphas for currentIndex and pairIndex
				double alphaCurrentOld = alphas[currentIndex];
				double alphaPairOld = alphas[pairIndex];
				// Calculating the bounds for the pairIndex's alpha
				double lowerBound, higherBound;
				if (this.trainingLabels[currentIndex] != this.trainingLabels[pairIndex]) {
					lowerBound = Math.max(0, alphas[pairIndex] - alphas[currentIndex]);
					higherBound = Math.min(this.cValue, this.cValue + alphas[pairIndex] - alphas[currentIndex]);
				} else {
					lowerBound = Math.max(0, alphas[pairIndex] + alphas[currentIndex] - this.cValue);
					higherBound = Math.min(this.cValue, alphas[pairIndex] + alphas[currentIndex]);
				}
				if (lowerBound == higherBound) {
					continue;
				}
				// Eta is a measure of the distance between two support vectors
				// Calculating eta, to determine how much to update alphas during optimisation
				double eta = 2 * RBFKernel(this.trainingData[currentIndex], this.trainingData[pairIndex])
						- RBFKernel(this.trainingData[currentIndex], this.trainingData[currentIndex]) - RBFKernel(this.trainingData[pairIndex], this.trainingData[pairIndex]);
				if (eta >= 0) {
					continue;
				}
				// Updating pairIndex's alpha
				alphas[pairIndex] -= this.trainingLabels[pairIndex] * (errorCurrent - errorPair) / eta;
				alphas[pairIndex] = Math.max(lowerBound, Math.min(higherBound, alphas[pairIndex]));
				// Checking if the update to alpha is significant
				if (Math.abs(alphas[pairIndex] - alphaPairOld) < this.tolerance) {
					continue;
				}
				// Updating currentIndex's alpha
				alphas[currentIndex] += this.trainingLabels[currentIndex] * this.trainingLabels[pairIndex] * (alphaPairOld - alphas[pairIndex]);
				// Updating bias terms
				double b1 = this.b - errorCurrent
						- this.trainingLabels[currentIndex] * (alphas[currentIndex] - alphaCurrentOld)
								* RBFKernel(this.trainingData[currentIndex], this.trainingData[currentIndex])
						- this.trainingLabels[pairIndex] * (alphas[pairIndex] - alphaPairOld) * RBFKernel(this.trainingData[currentIndex], this.trainingData[pairIndex]);
				double b2 = this.b - errorPair
						- this.trainingLabels[currentIndex] * (alphas[currentIndex] - alphaCurrentOld)
								* RBFKernel(this.trainingData[currentIndex], this.trainingData[pairIndex])
						- this.trainingLabels[pairIndex] * (alphas[pairIndex] - alphaPairOld) * RBFKernel(this.trainingData[pairIndex], this.trainingData[pairIndex]);
				if (0 < alphas[currentIndex] && alphas[currentIndex] < this.cValue) {
					this.b = b1;
				} else if (0 < alphas[pairIndex] && alphas[pairIndex] < this.cValue) {
					this.b = b2;
				} else {
					this.b = (b1 + b2) / 2;
				}
			}
		}
	}

	// Method to predict the label for a given instance
	public int predict(double[] instance) {
		double decision = 0.0;
		// Calculating the decision value using the SVM model considering only the support vectors
		for (int dataIndex = 0; dataIndex < trainingData.length; dataIndex++) {
			if (alphas[dataIndex] > 0) {
				decision += alphas[dataIndex] * trainingLabels[dataIndex] * RBFKernel(trainingData[dataIndex], instance);
			}
		}
		decision += this.b;// Adding the bias term
		return decision >= 0 ? 1 : -1;// Returning the label based on the decision value
	}

	// Method to calculate the error for a given instance
	private double calculateError(double[] instance, int label) {
		return calculateDecisionFunction(instance) - label;
	}

	// Method to calculate the decision for a given instance
	private double calculateDecisionFunction(double[] instance) {
		double decision = 0.0;
		// Calculating the decision value using the SVM model
		for (int dataIndex = 0; dataIndex < trainingData.length; dataIndex++) {
			if (alphas[dataIndex] > 0) {
				decision += alphas[dataIndex] * trainingLabels[dataIndex] * RBFKernel(trainingData[dataIndex], instance);
			}
		}
		decision += this.b;// Adding the bias term
		return decision;
	}

	// Method to execute the Radial Basis Function(RBF) kernel for two feature vectors
	private double RBFKernel(double[] firstFeatureVector, double[] secondFeatureVector) {
		double squaredDistance = Math.pow(euclideanDistance(firstFeatureVector, secondFeatureVector ), 2);
		return Math.exp(-this.gammaValue * squaredDistance);// Return the RBF kernel value
	}

	// Method to calculate the squared Euclidean distance between two feature vectors
	private double euclideanDistance(double[] firstFeatureVector, double[] secondFeatureVector) {
		double sumSquaredDifferences = 0.0;
		for (int valueIndex = 0; valueIndex < firstFeatureVector.length; valueIndex++) {
			sumSquaredDifferences += Math.pow(firstFeatureVector[valueIndex] - secondFeatureVector[valueIndex], 2);
		}

		return Math.sqrt(sumSquaredDifferences);// Return the euclidean distance
	}

}