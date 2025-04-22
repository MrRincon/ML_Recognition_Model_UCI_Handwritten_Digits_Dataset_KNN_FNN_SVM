//M00774667
package Coursework;

//Import any essential packages
import Coursework.algorithms.Hyperparameters;
import Coursework.data.Data_Loader;
import Coursework.validation.K_Fold_Cross_Validation;

public class Programme_Manager {
	private K_Fold_Cross_Validation bestCrossValidation;
	private String[] algorithms = { "SVM", "FNN", "KNN" };
	public void run() {
		// Initialisation of the data loader to access the data set
		Data_Loader loadFile = Data_Loader.getLoaderInstance();
		// Extracting the labels as a one-hot encode
		double[][] allOneHotLabels = loadFile.getAllOneHotEncodeLabels();
		System.out.println("Labels extracted and one-hot encoded successfully!");
		// Loading of all the normalised data
		double[][] dataFile = loadFile.getAllDataValuesNormalised();
		System.out.println("Data loaded and normalised successfully!");
		// Implementing different machine learning approaches
		for (String algorithm : this.algorithms) {
			// Tuning and storing of all possible hyperparameters for each algorithm
			Hyperparameters hyperparameters = new Hyperparameters(algorithm);
			double[][] hyperparameterCombinations = hyperparameters.getHyperparameterCombinations();
			System.out.println(getAlgorithmString(algorithm));
			double bestAccuracy = 0;
			// Implementation of the K-Fold Cross-Validation and FNN training for each hyperparameter combination
			for (double[] hyperparameterCombination : hyperparameterCombinations) {
				K_Fold_Cross_Validation kFoldCrossValidation = new K_Fold_Cross_Validation(dataFile, allOneHotLabels, hyperparameterCombination, algorithm);
				double currentAccuracy = kFoldCrossValidation.kFoldRun();
				// Maintaining the highest accuracy and the best hyperparameter combination stored
				if (currentAccuracy > bestAccuracy) {
					bestAccuracy = currentAccuracy;
					bestCrossValidation = kFoldCrossValidation;
				}
			}
			// Displaying the parameter values that achieved the best accuracy
			System.out.println("Best accuracy achieved...");
			System.out.println(this.bestCrossValidation.toString());
			this.bestCrossValidation.getConfusionMatrix();
		}

	}

	// Method to return a formatted string to specify the machine learning system currently operating
	private String getAlgorithmString(String algorithm) {
		int boxWidth = 115;
		StringBuilder sb = new StringBuilder();
		sb.append("=".repeat(boxWidth)).append("\n");
		String title = String.format("Evaluating all hyperparameters combinations for the best accuracy with %s", algorithm);
		int padding = (boxWidth - 2 - title.length()) / 2;
		sb.append("|").append(" ".repeat(padding)).append(title).append(" ".repeat(padding));
		if ((boxWidth - 2 - title.length()) % 2 != 0) {
			sb.append(" ");
		}
		sb.append("|\n");
		sb.append("=".repeat(boxWidth)).append("\n");

		return sb.toString();
	}
}