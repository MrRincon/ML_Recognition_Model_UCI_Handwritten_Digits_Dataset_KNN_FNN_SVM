//M00774667
package Coursework.algorithms;

//Import any essential packages
import java.util.Arrays;
import java.util.Random;

import Coursework.validation.Confusion_Matrix;

public class Feedforward_Neural_Network {
	private int inputSize; // Number of input labels
	private int hiddenSize; // Number of hidden neurons
	private int outputSize; // Number of output neurons
	private double learningRate;// Learning rate for weight updates
	private int epochs; // number of epochs
	private double[][] weightsInputHidden; // Weights between input and hidden layer
	private double[][] weightsHiddenOutput; // Weights between hidden and output layer

	// Constructor to initialise the neural network parameters
	public Feedforward_Neural_Network(double[] hyperparameters) {
		this.inputSize = 64;
		this.outputSize = 10;
		this.hiddenSize = (int) hyperparameters[0];
		this.learningRate = hyperparameters[1];
		this.epochs = (int) hyperparameters[2];
		this.weightsInputHidden = new double[this.inputSize][this.hiddenSize];
		this.weightsHiddenOutput = new double[this.hiddenSize][this.outputSize];
		initialiseWeights();
	}

	// Method to initialise weights with small random values
	private void initialiseWeights() {
		Random rand = new Random();
		// Initialise weights from input to hidden layer
		for (int inputNeuron = 0; inputNeuron < this.inputSize; inputNeuron++) {
			for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
				this.weightsInputHidden[inputNeuron][hiddenNeuron] = rand.nextDouble() * 0.01;
			}
		}
		// Initialise weights from hidden to output layer
		for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
			for (int outputNeuron = 0; outputNeuron < this.outputSize; outputNeuron++) {
				this.weightsHiddenOutput[hiddenNeuron][outputNeuron] = rand.nextDouble() * 0.01;
			}
		}
	}

	// Training method for the neural network
	public void train(double[][] trainingData, double[][] trainingNeurons) {
		// Looping through each epoch for training
		for (int epoch = 0; epoch < this.epochs; epoch++) {
			for (int trainingIndex = 0; trainingIndex < trainingData.length; trainingIndex++) {
				// Applying back propagation for each training sample
				backPropagation(trainingData[trainingIndex], trainingNeurons[trainingIndex]);
			}
		}
	}

	// Testing method to evaluate model's performance
	public double test(double[][] testData, double[][] testNeurons, Confusion_Matrix confusionMatrix) {
		int correctPredictions = 0;
		// Looping through each test data sample
		for (int testDataIndex = 0; testDataIndex < testData.length; testDataIndex++) {
			int predictedNeuron = predict(testData[testDataIndex]);
			int actualNeuron = getNeuronFromOneHot(testNeurons[testDataIndex]);
			confusionMatrix.updateMatrix(actualNeuron, predictedNeuron);// Updating the confusion matrix with the actual and predicted neuron
			if (predictedNeuron == actualNeuron) {
				correctPredictions++;// Counting the correct predictions
			}
		}

		return (double) correctPredictions / testData.length * 100;// Calculating the accuracy
	}

	// Prediction method for a given input sample
	public int predict(double[] input) {
		double[] output = forwardPropagation(input)[1];// Performing forward propagation to get output layer values
		int predictedNeuron = 0;
		// Looping to find the index of the highest output value
		for (int outputIndex = 1; outputIndex < output.length; outputIndex++) {
			if (output[outputIndex] > output[predictedNeuron]) {
				predictedNeuron = outputIndex;
			}
		}

		return predictedNeuron;// Returns the index of the predicted neuron
	}

	// Method to extract the neuron from the one-hot encoded array
	private int getNeuronFromOneHot(double[] oneHotArray) {
		for (int neuronIndex = 0; neuronIndex < oneHotArray.length; neuronIndex++) {
			if (oneHotArray[neuronIndex] == 1.0) {
				return neuronIndex;
			}
		}

		return -1;// Returns -1 if no active neuron is found
	}

	// Back propagation method to adjust weights based on the error
	public void backPropagation(double[] input, double[] target) {
		// Performing forward propagation to get activations
		double[][] activations = forwardPropagation(input);
		double[] hiddenLayerActivations = activations[0];
		double[] outputLayerActivations = activations[1];
		// Calculating output layer error
		double[] outputErrors = new double[this.outputSize];
		for (int outputNeuron = 0; outputNeuron < this.outputSize; outputNeuron++) {
			outputErrors[outputNeuron] = target[outputNeuron] - outputLayerActivations[outputNeuron];
		}
		// Updating weights from hidden to output layer based on the output error
		for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
			for (int outputNeuron = 0; outputNeuron < this.outputSize; outputNeuron++) {
				this.weightsHiddenOutput[hiddenNeuron][outputNeuron] += this.learningRate * outputErrors[outputNeuron]
						* hiddenLayerActivations[hiddenNeuron];
			}
		}
		// Calculating hidden layer error using output errors and weights
		double[] hiddenErrors = new double[this.hiddenSize];
		for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
			hiddenErrors[hiddenNeuron] = 0;
			for (int outputNeuron = 0; outputNeuron < this.outputSize; outputNeuron++) {
				hiddenErrors[hiddenNeuron] += outputErrors[outputNeuron]
						* this.weightsHiddenOutput[hiddenNeuron][outputNeuron];
			}
			hiddenErrors[hiddenNeuron] *= reluDerivative(hiddenLayerActivations[hiddenNeuron]); // Applying sigmoid derivative
		}
		// Updating weights from input to hidden layer based on the hidden error
		for (int inputNeuron = 0; inputNeuron < this.inputSize; inputNeuron++) {
			for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
				this.weightsInputHidden[inputNeuron][hiddenNeuron] += this.learningRate * hiddenErrors[hiddenNeuron]
						* input[inputNeuron];
			}
		}
	}

	// Forward propagation method to compute activations of hidden and output layers
	public double[][] forwardPropagation(double[] input) {
		double[] hiddenLayerActivations = new double[this.hiddenSize];
		double[] outputLayerActivations = new double[this.outputSize];
		// Looping to calculate hidden layer activations using input and weights
		for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
			hiddenLayerActivations[hiddenNeuron] = 0;
			for (int inputNeuron = 0; inputNeuron < this.inputSize; inputNeuron++) {
				hiddenLayerActivations[hiddenNeuron] += input[inputNeuron]
						* this.weightsInputHidden[inputNeuron][hiddenNeuron];
			}
			hiddenLayerActivations[hiddenNeuron] = relu(hiddenLayerActivations[hiddenNeuron]);// Applying sigmoid activation
		}
		// Looping to calculate output layer activations using hidden layer and weights
		for (int outputNeuron = 0; outputNeuron < this.outputSize; outputNeuron++) {
			outputLayerActivations[outputNeuron] = 0;
			for (int hiddenNeuron = 0; hiddenNeuron < this.hiddenSize; hiddenNeuron++) {
				outputLayerActivations[outputNeuron] += hiddenLayerActivations[hiddenNeuron]
						* this.weightsHiddenOutput[hiddenNeuron][outputNeuron];
			}
		}
		// Applying softmax function to the output layer activations to normalised to probabilities
		outputLayerActivations = softmax(outputLayerActivations);

		return new double[][] { hiddenLayerActivations, outputLayerActivations };// Returns 2D array with hidden and output layer activations
	}

	// ReLU activation function
	private double relu(double x) {
	    return Math.max(0, x);
	}

	// Derivative of ReLU for backpropagation
	private double reluDerivative(double x) {
	    return x > 0 ? 1 : 0;
	}

	// Softmax function to normalise the output layer values to probabilities
	private double[] softmax(double[] output) {
		double[] softmaxOutput = new double[output.length];
		double max = Arrays.stream(output).max().getAsDouble();// Finding max value for numerical stability
		double sum = 0.0;
		// Looping to exponentiate each output layer activation(offset by max for stability) and compare their sum
		for (int outputIndex = 0; outputIndex < output.length; outputIndex++) {
			softmaxOutput[outputIndex] = Math.exp(output[outputIndex] - max);
			sum += softmaxOutput[outputIndex];
		}
		// Looping to normalise each exponentiated value by dividing by the sum to get probabilities
		for (int outputIndex = 0; outputIndex < softmaxOutput.length; outputIndex++) {
			softmaxOutput[outputIndex] /= sum;
		}

		return softmaxOutput;// Returns the probabilities of the output layer
	}
}