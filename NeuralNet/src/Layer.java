import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

/**
 * Class Layer is superclass containing the types of layers in a neural network
 * The neural network is divided into abstract layer objects which contain
 * information about the layer itself which is used for calculation
 */
public class Layer {
	// The position of the layer is contained in the layerPosition val
	int layerPosition;

	// The matrix of "final" layerValues is stored in layerValue
	double[][] layerValue;
	// currentBatch holds the batch that is currently being operated on in the
	// network
	double[][] currentBatch;
	// holds the preActivatedValue of a layer used for backpropagation
	double[][] preActivatedValue;
	// If the amount of data is above a threshold then it is split into training and
	// testing data. The training data is stored in layerValue
	// while testData is stored in testData
	double[][] testData;

	// Each layer has a weightValue associated with it (except for the output layer)
	// because each layer is operated on
	// using a weightValue
	double[][] weightValue;

	// stores the number of nodes in a layer (columns)
	int layerSize;

	// random seed for shuffling input data the same way in two arrays
	final static long seed = (long) (1 + (new Random().nextFloat() * (10000 - 1)));
	// stores the activation associated with each layer
	String activation;

	/**
	 * parses the type of object into an integer associated with that type of layer
	 * used for model saving numbers are nonsequential so older data can still be
	 * used
	 * 
	 * @param layer the layer to be parsed
	 * @return returns the associated integer
	 */
	public static int parseObjectTypeIntoInt(Layer layer) {
		int output = 0;
		if (layer instanceof InputLayer) {
			output = 0;
		} else if (layer instanceof HiddenLayer) {
			output = 1;
		} else if (layer instanceof OutputLayer) {
			output = 6;
		}
		return output;
	}

}

/**
 * Class InputLayer is a subclass of Layer and is an abstract representation of
 * an input layer which is required in a every type of network it is also the
 * most complicated type of layer because it handles the data preprocessing as
 * well
 * 
 * 
 *
 */
class InputLayer extends Layer {
	// numofSets represents the number of sets of data i.e. the numner of games
	// played
	int numofSets = 0;
	// numofInput represents the number of stats within each stat
	int numofInput = 0;
	// batchSize represents the num of numofSets processed at one time
	int batchSize = 0;
	// to create batches of size batch size there may be a leftover batch smaller
	// than batchSize, the size of that is stored in
	// remainingBatchSize
	int remainingBatchSize = 0;
	// targetSize refers to the size of the result of a "game" i.e. the final score
	// (in which case the size would be 2 one for each team)
	int targetSize;
	// numofBatches holds the number of batches resulting from a batch split
	static int numofBatches = 0; // possibly find a way to change away from static
	// fileName holds the string of the file name which holds the input data
	String fileName = "";
	// the strdFilePath holds the stardard file path of a user for easier management
	// of the file system
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\NeuralNetworkRelated\\";

	// fileReader object of class FileReader
	FileReader fileReader;
	// normalizer object of class normalizer to normalize the input data
	Normalizer normalizer = new Normalizer();
	// holds a list of batch objects which holds each input batch as batch objects
	List<Batch> batchList = new ArrayList<Batch>();

	Targets targets = new Targets();

	/**
	 * Constructor which creates an InputLayer object with numofSets, the numofInput
	 * present in the file, the batchSize to be used and where the file can be
	 * accessed the remaining batchSize is calculated by modulating the numofSets
	 * with the batchSize the fileReader is assigned to the correct file using the
	 * strdFilePath
	 */
	public InputLayer(int numofSets, int numofInput, int batchSize, String fileName) {
		layerSize = numofInput;
		this.numofSets = numofSets;
		this.numofInput = numofInput;
		this.batchSize = batchSize;
		this.fileName = fileName;
		remainingBatchSize = (numofSets % batchSize);
		fileReader = new FileReader(strdFilePath + fileName + ".txt");
	}

	/**
	 * initializeLayer is used for data preprocessing in a "normal" situation. That
	 * is, this method is called when the data is arranged as CSV (with no commas)
	 * 
	 * the information is parsed from the file into arrays of proper size for
	 * operation and then the rows are shuffled to eliminate and statistical bias.
	 * Then the data is separated into a training and testing set and normalized
	 * accordingly then finally the data is formatted to a type which can be used
	 * for training
	 * 
	 */
	public void initializeLayer() { // add error handling
		parseInformationFromFile();

		shuffleArray(layerValue);

		trainTestSplitNormalize();

		formatInput();
	}

	/**
	 * initializeDiagnosticLayer is used preprocess the data still in a normal
	 * situation however in this case the data never changes
	 *
	 * the information is parsed from the file into arrays of proper size for
	 * operation. Then the data is separated into a training and testing set and
	 * normalized accordingly then finally the data is formatted to a type which can
	 * be used for training
	 */
	public void initializeDiagnosticLayer() { // does not shuffle the inputs
		parseInformationFromFile();

		trainTestSplitNormalize();

		formatInput();
	}

	/**
	 * initializeLayerText is used to preprocess information in a normal situation
	 * except for the data is presented in text files as abstract text representing
	 * the data
	 * 
	 * the information is first parsed from the lookup table into arrays so that it
	 * may be processed in the same way
	 * 
	 * TODO why targets.targetSize must be assigned to target size
	 * 
	 * the array is shuffled to stop any bias with data entering Then the data is
	 * separated into a training and testing set and normalized accordingly then
	 * finally the data is formatted to a type which can be used for training
	 * 
	 * @param lookup
	 */
	public void initializeLayerText(String lookup) { // add error handling
		layerValue = (fileReader.parseInputIntoArray(numofSets, numofInput, lookup));

		targetSize = fileReader.determineTargetSizeWithText(numofSets);

		targets.targetSize = targetSize;

		shuffleArray(layerValue);

		trainTestSplitNormalize();

		formatInput();
	}

	/**
	 * Parsing information from the file is used several places and thus can broken
	 * into an abstraction first the layerValue of the layer gets fed the input from
	 * the file into an array using a separate method the target size is determined
	 * from the file information and assigned appropriately
	 */
	private void parseInformationFromFile() {
		layerValue = (fileReader.readInputIntoArray(numofSets, numofInput));
		targetSize = fileReader.determineTargetSize(numofSets, numofInput);
		targets.targetSize = targetSize;
	}

	/**
	 * This (subroutine) is used in several cases and is thus abstractionalized into
	 * a different method If the train test split threshold is met then data sets
	 * are split And the training set is normalized to z scores. And then the
	 * testing set is normalized to the same parameters as the training set
	 * 
	 * if the threshold isn't met then the layerValue is normalized
	 * 
	 * 
	 */
	private void trainTestSplitNormalize() {
		if (Utility.testSplitThreshold(this)) { // TODO check if this works
			trainTestSplit();
			layerValue = normalizer.normalizeInputsZscore(layerValue, targetSize);
			initializeTestData();
			testData = normalizer.normalizeInputs(testData, normalizer.meanArray, normalizer.strdDev);
		} else {
			layerValue = normalizer.normalizeInputsZscore(layerValue, targetSize);
		}
	}

	/**
	 * This (subroutine) is used in several cases and is thus abstractionalized into
	 * a different method The targets are assigned to the target object using a
	 * separate method the actual values are separated are separated from the
	 * targets (which are also normalized) then a bias column is added to those
	 * values
	 * 
	 * And finally the training set is split into training batches
	 */
	private void formatInput() {
		targets.determineTargets(layerValue, numofInput);
		layerValue = extractInputs(layerValue);
		layerValue = Utility.appendBiasColumn(layerValue);

		splitIntoBatches();
	}

	/**
	 * calculateNumofBatches simply uses the number of rows of the layerValue and
	 * divides that by the batch size to get the total number of batches used in
	 * training. Ceil is used to round up whatever division because if it doesn't
	 * divide equally a smaller batch is made for the leftover set (hence the extra
	 * batch/round up)
	 * 
	 * @return returns the result of the division
	 */
	private int calculateNumofBatches() {
		return (int) Math.ceil((double) layerValue.length / batchSize);
	}

	/**
	 * first calculates the num of batches to be split into (using the above method)
	 * and then creates a batch object for each necessary batch to be created using
	 * a for loop which assigns each batch object a batch value
	 */
	private void splitIntoBatches() {
		numofBatches = calculateNumofBatches();
		Batch batch;

		// loops through all the batches except for the last batch
		for (int i = 0; i < numofBatches - 1; i++) {
			batch = new Batch();
			getBatch(batch); // rename
		}

		batch = new Batch();
		getBatchRemaining(batch); // rename
	}

	int batchCounter = 0;

	/**
	 * Extracts a small subset of layerValue into a batch which is assigned to an
	 * object
	 * 
	 * getBatch() is called by a loop and calls each batch except for the remaining
	 * batch which is handled by the subsequent method Nested for loops then extract
	 * information from layerValue (of the inputLayer) using a batchCounter to track
	 * where in the array to be extracting information
	 * 
	 * the batchValue is assigned to the object and the increment is increased
	 * 
	 * @param batch: the batch object to be assigned to
	 */
	private void getBatch(Batch batch) { // rename
		double[][] batchVal = new double[batchSize][layerValue[0].length];

		for (int i = 0; i < batchSize; i++) {
			for (int j = 0; j < layerValue[0].length; j++) { // added one for the bias column
				batchVal[i][j] = layerValue[i + batchCounter * batchSize][j];
			}
		}
		batch.batchValue = batchVal;
		batchCounter++;

		batchList.add(batch);
	}

	/**
	 * First checks if the remaining batch size is 0 which means the batches are
	 * divided up evenly if it is 0 then the remaining batch size is set to a full
	 * batch size
	 * 
	 * Nested for loops then extract information from layerValue (of the inputLayer)
	 * using a batchCounter to track where in the array to be extracting information
	 * 
	 * the batchValue is assigned to the object and the finalBatch flagged is
	 * assinged to the batch object
	 * 
	 * @param batch: the batch object to be assigned to
	 */
	private void getBatchRemaining(Batch batch) { // rename
		double[][] batchVal;

		if (remainingBatchSize == 0) {
			remainingBatchSize = batchSize;
		}

		batchVal = new double[remainingBatchSize][layerValue[0].length];
		for (int i = 0; i < remainingBatchSize; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				batchVal[i][j] = layerValue[i + batchCounter * batchSize][j];
			}
		}
		batch.batchValue = batchVal;
		batch.finalBatch = true;

		batchList.add(batch);
	}

	int trainingSize;

	/**
	 * Splits the data into a training set and a test set
	 * The training size is set to 70% of the total data and the testing to 30% 
	 * 
	 * Using nested for loops the 70% of the data is extracted into trainingData and 30% to testing data
	 * The layerValue is then set to this training data
	 * 
	 */
	private void trainTestSplit() {
		double[][] trainingData;
		trainingSize = (int) (.7 * numofSets);
		remainingBatchSize = (trainingSize % batchSize); // used to recalculate the remaining batch size if the
															// traintestsplit condition is met

		int testingSize = numofSets - trainingSize;

		trainingData = new double[trainingSize][numofInput + targetSize];
		testData = new double[testingSize][numofInput + targetSize];

		for (int i = 0; i < trainingSize; i++) {
			for (int j = 0; j < numofInput + targetSize; j++) {
				trainingData[i][j] = layerValue[i][j];
			}
		}
		for (int i = trainingSize; i < numofSets; i++) {
			for (int j = 0; j < numofInput + targetSize; j++) {
				testData[i - trainingSize][j] = layerValue[i][j];
			}
		}

		layerValue = trainingData;
	}

	/**
	 * This subroutine (method) is used in several places and is thus condensed into a method for repetitive use 
	 * First the targets for the testing data is determined and then the test data is prepared by extraction and subsequently a bias column 
	 */
	private void initializeTestData() {
		targets.determineTestTargets(testData, numofInput, trainingSize);
		testData = extractInputs(testData);
		Utility.appendBiasColumn(testData);
	}

	/**
	 * ShuffleArray is used to shuffle the rows of an array randomly 
	 * First the two dimensional array is reformatted into a list of single dimensional arrays 
	 * Then using the collections shuffle method the list is shuffled 
	 * Then the list of single dimensional arrays is reformatted as a two dimensional array 
	 * 
	 * @param inputLayer: the inputLayer's layerValue to shuffle (each set) 
	 */
	private void shuffleArray(double[][] inputLayer) {
		double[] array;
		List<double[]> list = new ArrayList<double[]>(inputLayer.length);

		for (int i = 0; i < inputLayer.length; i++) {
			array = new double[inputLayer[0].length];
			for (int j = 0; j < inputLayer[0].length; j++) {
				array[j] = inputLayer[i][j];
			}
			list.add(array);
		}

		Collections.shuffle(list);

		for (int i = 0; i < inputLayer.length; i++) {
			for (int j = 0; j < inputLayer[0].length; j++) {
				inputLayer[i][j] = list.get(i)[j];
			}
		}
	}

	/**
	 * Maps the inputs from a layerValue with targets included into a separate array without the targets 
	 * 
	 * @param inputs: the inputs from a layerValue with the targets
	 * @return returns the extracted inputs 
	 */
	private double[][] extractInputs(double[][] inputs) {
		double[][] result = new double[inputs.length][inputs[0].length - targetSize];

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[0].length - targetSize; j++) {
				result[i][j] = inputs[i][j];
			}
		}
		return result;
	}
} // end of class inputlayer

/**
 * Class Batch holds batch objects as part for the input layer
 *
 */
class Batch {

	double[][] batchValue;
	boolean finalBatch = false;
}

/**
 * Class HiddenLayer holds all dense network layers between the input and output Layer
 *
 */
class HiddenLayer extends Layer {
	int numofNeuron = 0; //TODO needed? already held in layerSize? 

	/**
	 * Constructs a hiddenLayer object
	 * @param numofNeuron: the number of hidden neurons in the layer
	 * @param activation: the activation associated layer (which activation type the layer gets activated with) 
	 */
	public HiddenLayer(int numofNeuron, String activation) {
		layerSize = numofNeuron;
		this.numofNeuron = numofNeuron;
		this.activation = activation;
	}

}

/**
 * Class OutputLayer stores the information of the output layer of the network
 * Each network can only have one output layer 
 * The only difference between a dense hidden layer and the output layer is the output layers association with the targets 
 * and lack of association with weights
 *
 */
class OutputLayer extends Layer {
	int numofOutputNeuron = 0; //needed? TODO 
	String targetFile;

	/**
	 * Constructor outputLayer constructs an output layer object with the activation type and the number of output neurons (the size of the targets) 
	 * @param numofOutputNeuron
	 * @param activation
	 */
	public OutputLayer(int numofOutputNeuron, String activation) {
		layerSize = numofOutputNeuron;
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation;
	}
	
	/**
	 * This constructor of output layer also creates an outputLayer with number of output neurons and activation as well but this constructor is 
	 * used when the targets are stored in another file instead of the same file as the input
	 * @param numofOutputNeuron
	 * @param activation
	 * @param targetFile
	 */
	public OutputLayer(int numofOutputNeuron, String activation, String targetFile) {
		layerSize = numofOutputNeuron;
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation;
		this.targetFile = targetFile;
	}

}

/**
 * Class dropoutLayer creates a dropoutLayer object
 * Dropout layers differ from dense layers because random neurons will be set to 0 so the network is forced to use new pathways to solve information
 * The neurons are  "dropped out" proportional to the dropoutProbability
 *
 */
class DropoutLayer extends Layer {
	int numofNeuron = 0;
	double dropoutProbability = 0.0;

	/**
	 * Constructor dropoutProbability creates a dropoutLayer with dropout Probability: dropoutProbability
	 * as well as number of neurons (without dropout) and activation 
	 * @param numofNeuron
	 * @param dropoutProbability
	 * @param activation
	 */
	public DropoutLayer(int numofNeuron, double dropoutProbability, String activation) {
		layerSize = numofNeuron;
		this.numofNeuron = numofNeuron;
		this.activation = activation;
		this.dropoutProbability = dropoutProbability;
	}
}
