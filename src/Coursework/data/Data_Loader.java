//M00774667
package Coursework.data;

//Import any essential packages
import java.io.File;
import java.io.FileNotFoundException;
import java.util.Arrays;
import java.util.Scanner;

public class Data_Loader {
	private static Data_Loader instance;// Singleton instance of DataLoader
	private final String filePath = System.getProperty("user.dir");// File path to current directory
	private final String firstDataFile = filePath + File.separator + "dataSet1.csv";// Path to the first CSV file
	private final String secondDataFile = filePath + File.separator + "dataSet2.csv";// Path to the second CSV file
	private static final int NUMBER_OF_ROWS = 2810;// Constant for the number of rows per file
	private static final int NUMBER_OF_COLUMNS = 65;// Constant for the number of columns per file
	private final int[][] allDataValues;

	// Private constructor for singleton pattern
	private Data_Loader() {
		this.allDataValues = new int[NUMBER_OF_ROWS * 2][NUMBER_OF_COLUMNS];
		readFiles();
	}

	// Method to get the singleton instance of DataLoader
	public static Data_Loader getLoaderInstance() {
		if (instance != null) {
			return instance;
		} else {
			instance = new Data_Loader();
			return instance;
		}
	}

	// Method to read training and testing files
	private void readFiles() {
		int rowsRead = readCSV(firstDataFile, allDataValues, 0);
		readCSV(secondDataFile, allDataValues, rowsRead);
	}

	// Method to read the CSV file and populate the data array starting from a specific row
	private int readCSV(String filename, int[][] dataArray, int startRow) {
		File file = new File(filename);
		int currentRow = startRow;// Variable to keep track of the current row to populate
		try (Scanner scanner = new Scanner(file)) {
			// Looping through each line in the file
			while (scanner.hasNextLine() && currentRow < dataArray.length) {
				String line = scanner.nextLine();
				String[] values = line.split(",");
				// Ensures that the row has the expected number of columns
				if (values.length != NUMBER_OF_COLUMNS) {
					System.out.println(
							"Error: Unexpected number of columns in file: " + filename + " at row " + currentRow);
					continue; // Skips rows with incorrect amount of columns
				}
				// Parsing of each value and populate the current row
				for (int currentColumn = 0; currentColumn < NUMBER_OF_COLUMNS; currentColumn++) {
					try {
						dataArray[currentRow][currentColumn] = Integer.parseInt(values[currentColumn]);
					} catch (NumberFormatException e) {
						System.out.println("Error parsing number at row " + currentRow + " in file: " + filename);
						continue; // Skips rows with parsing errors
					}
				}
				currentRow++;
			}
		} catch (FileNotFoundException e) {
			System.out.println("File not found, The csv files need to be in the same folder as the src folder." + filename);// Handling of a file not found
			System.exit(0);
		}

		return currentRow;// Returns the number of rows successfully read
	}

	// Method to extract labels from all the data
	public double[][] getAllOneHotEncodeLabels() {
		int[] labelsDeepCopy = new int[allDataValues.length];
		// Deep copy of labels from the last column of allDataValues
		for (int rowIndex = 0; rowIndex < allDataValues.length; rowIndex++) {
			// Extracting the label present at the end of each row and removing it from the data
			labelsDeepCopy[rowIndex] = allDataValues[rowIndex][allDataValues[rowIndex].length - 1];
			allDataValues[rowIndex] = Arrays.copyOf(allDataValues[rowIndex], allDataValues[rowIndex].length - 1);
		}

		return oneHotEncodeLabels(labelsDeepCopy);// Returns the labels as a one-hot encode
	}

	// Method to one-hot encode labels
	private double[][] oneHotEncodeLabels(int[] labels) {
		double[][] oneHotLabels = new double[labels.length][10];
		// Setting the corresponding index in each row of oneHotLabels to 1.0 based on label
		for (int labelIndex = 0; labelIndex < labels.length; labelIndex++) {
			oneHotLabels[labelIndex][labels[labelIndex]] = 1.0;
		}

		return oneHotLabels;
	}

	// Method to get a normalised deep copy of all the data in the array (first data and second data)
	public double[][] getAllDataValuesNormalised() {
		int[][] dataDeepCopy = new int[allDataValues.length][NUMBER_OF_COLUMNS - 1];
		// Deep copy of data without labels for normalisation
		for (int rowIndex = 0; rowIndex < allDataValues.length; rowIndex++) {
			System.arraycopy(allDataValues[rowIndex], 0, dataDeepCopy[rowIndex], 0, NUMBER_OF_COLUMNS - 1);
		}

		return normaliseData(dataDeepCopy, dataDeepCopy.length);// Returns normalised data
	}

	// Method to normalise the pixel data values into a range [0, 1]
	private double[][] normaliseData(int[][] dataArray, int rowCount) {
		double[][] normalisedData = new double[rowCount][NUMBER_OF_COLUMNS - 1];
		// Normalising each value by dividing by 255
		for (int rowIndex = 0; rowIndex < rowCount; rowIndex++) {
			for (int colIndex = 0; colIndex < NUMBER_OF_COLUMNS - 1; colIndex++) {
				normalisedData[rowIndex][colIndex] = dataArray[rowIndex][colIndex] / 255.0;
			}
		}

		return normalisedData;// Returns normalised data array
	}
}