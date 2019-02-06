import java.io.File;
import java.util.ArrayList;
import java.util.List;

public class NetworkEvaluator {
	Normalizer normalizer = new Normalizer();
	NetworkTrainer nt = new NetworkTrainer();
	Activator activator = new Activator();
	FileReader fr;
	List<Double> listOfValues = new ArrayList<Double>();
	List<double[][]> weightList = new ArrayList<double[][]>();
	List<Filters> filterList = new ArrayList<Filters>();
	List<double[][]> layerList = new ArrayList<double[][]>();
	List<double[][][]> imageList = new ArrayList<double[][][]>();
	List<Layer> layerListObjects = new ArrayList<Layer>();
	Weights weights = new Weights(); 
	String filePath = System.getProperty("user.home") + "\\Desktop\\Models\\";
	String testFilePath = System.getProperty("user.home") + "\\Desktop\\";
	Layer layer = new Layer();
	int numofLayers;
	int[] layerSizes;
	int[] layerTypes;
	double[] mean;
	double[] strdDev;
	String[] activationStrings;
	double[][] weightArray;
	double[][] inputLayer;
	double[][] layerValue;

	public void predict(String model, double... inputs) {
		inputLayer = new double[1][inputs.length];

		acquireModelValues(model);

		for (int i = 0; i < inputs.length; i++) {
			inputLayer[0][i] = inputs[i];
		}

		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();

		listOfValues.clear();

		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue) + "\n");

	} // end of predict

	public void predict(String modelFilePath, String testFilePath) {

		acquireModelValues(modelFilePath);
		acquireTestValues(testFilePath);
		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();

		listOfValues.clear();

		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue) + "\n");
	}

	public void predictConv(String modelFilePath, String testFilePath) {
		acquireConvModelValues(modelFilePath);
		acquireConvTestValues(testFilePath);
		
		weights.weightList = weightList;
		weights.filterList = filterList;
		
		System.out.println(filterList.get(0).threeDFilterArray.get(0).length);
		
		nt.fp.constructForwardPropagationObjects(layerListObjects, weights);
		
		forwardPropagationConv(); 

		listOfValues.clear();

		System.out.println("Prediction " + java.util.Arrays.deepToString(layerListObjects.get(layerListObjects.size()-1).layerValue) + "\n");
	}

	private void formWeightsToArrays(int[] layerSizes) {
		int counter = 0;
		for (int k = 0; k < layerSizes.length - 1; k++) {
			weightArray = new double[layerSizes[k] + 1][layerSizes[k + 1]];
			for (int i = 0; i < layerSizes[k] + 1; i++) {
				for (int j = 0; j < layerSizes[k + 1]; j++) {
					weightArray[i][j] = listOfValues.get(counter);
					counter++;
				}
			}
			weightList.add(weightArray);
		}
	}

	public void forwardPropagation() {
		propagateInputLayer();
		for (int i = 1; i < numofLayers - 1; i++) {
			layerValue = appendBiasColumn(layerValue);
			layerValue = nt.matrixMultiplication(layerValue, weightList.get(i));
			layer.layerValue = layerValue;
			layer.activation = activationStrings[i];
			layerValue = nt.activate(layer);
		}
	}

	public void forwardPropagationConv() {
		((ConvolutionalLayer) layerListObjects.get(0)).trainingImages = imageList;
	
		long start = System.nanoTime();
		for (int i = 0; i < layerListObjects.size() - 1; i++) {
			layerListObjects.get(i + 1).layerValue = nt.fp.propagate(layerListObjects.get(i), layerListObjects.get(i + 1));
		}
		long end = System.nanoTime();
		
		System.out.println(nt.getTrainingTime(start, end));
		
	}
	
	public void propagateInputLayer() {
		layerValue = appendBiasColumn(inputLayer);
		System.out.println("InitialLayer " + java.util.Arrays.deepToString(layerValue));
		layerValue = nt.matrixMultiplication(layerValue, weightList.get(0));
		layer.layerValue = layerValue;
		layer.activation = activationStrings[0];
		layerValue = nt.activate(layer);
	}

	public void acquireModelValues(String modelPath) {
		fr = new FileReader(filePath + modelPath + ".txt");
		fr.initializeBufferedReader(); // initializes scanner
		fr.readDataIntoList();

		listOfValues = fr.valuesFromFile;
		numofLayers = (int) (double) listOfValues.get(0);

		listOfValues.remove(0); // remove numofLayers

		int numofSets = (int) (double) listOfValues.get(0);

		layerSizes = new int[numofLayers];
		mean = new double[numofSets];
		strdDev = new double[numofSets];

		activationStrings = new String[numofLayers - 1];

		for (int i = 0; i < numofLayers; i++) {
			layerSizes[i] = (int) (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		for (int i = 0; i < numofLayers - 1; i++) {
			activationStrings[i] = activator.convertActivationInt((int) (double) listOfValues.get(0));
			listOfValues.remove(0);
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			mean[i] = (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			strdDev[i] = (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		formWeightsToArrays(layerSizes);
	}

	int layerSize = 0;
	int inputHeight;
	int inputWidth;
	int inputChannels;
	int numofFilters;
	int filterSize;
	int strideLength;
	int poolSize;
	String padding;

	String activation = null;

	public void acquireConvModelValues(String modelPath) {
		fr = new FileReader(filePath + modelPath + ".txt");
		fr.initializeFileReader(); // initializes scanner
		fr.readFileIntoList();

		listOfValues = fr.valuesFromFile;
		numofLayers = (int) (double) listOfValues.get(0);
		listOfValues.remove(0); // remove numofLayers

		

		layerTypes = new int[numofLayers];

		for (int i = 0; i < numofLayers; i++) {
			layerTypes[i] = (int) (double) listOfValues.get(0);
			listOfValues.remove(0);
		}

		for (int i = 0; i < numofLayers; i++) { // layerTypes[0] is InputLayer, will change upon refactoring
			if (layerTypes[i] == 1) { // hidden
				layerSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				activation = activator.convertActivationInt((int) (double) listOfValues.get(0));
				listOfValues.remove(0);
				HiddenLayer hiddenLayer = new HiddenLayer(layerSize, activation);
				layerListObjects.add(hiddenLayer);
				
			} else if (layerTypes[i] == 2) { // conv text
				inputHeight = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				inputWidth = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				inputChannels = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				numofFilters = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				filterSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				strideLength = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				padding = "zero-padding"; // have to change if more types later
				listOfValues.remove(0);
				ConvolutionalLayer convLayer = new ConvolutionalLayer(inputHeight, inputWidth, inputChannels,
						numofFilters, filterSize, strideLength, padding);
				layerListObjects.add(convLayer);
				
			} else if (layerTypes[i] == 3) { //hiddenConv
				numofFilters = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				filterSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				strideLength = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				padding = "zeroPadding";
				listOfValues.remove(0);
				HiddenConvolutionalLayer hidden = new HiddenConvolutionalLayer(numofFilters, filterSize, strideLength,
						padding);
				layerListObjects.add(hidden);
				
			} else if (layerTypes[i] == 4) { //pool
				poolSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				PoolingLayer layer = new PoolingLayer(poolSize, "MAX");
				layerListObjects.add(layer);
				
			} else if (layerTypes[i] == 5) { //relu
				ReluLayer relu = new ReluLayer();
				layerListObjects.add(relu);
				
			} else if (layerTypes[i] == 6) { //output
				layerSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				activation = activator.convertActivationInt((int) (double) listOfValues.get(0));
				listOfValues.remove(0);
				OutputLayer outputLayer = new OutputLayer(layerSize, activation);
				layerListObjects.add(outputLayer);
				
			} else if (layerTypes[i] == 7) { //conv image
				numofFilters = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				filterSize = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				strideLength = (int) (double) listOfValues.get(0);
				listOfValues.remove(0);
				padding = "zero-padding"; // have to change if more types later
				listOfValues.remove(0);
				ConvolutionalLayer conv = new ConvolutionalLayer(numofFilters, filterSize, strideLength, padding);
				conv.channelDepth = 3; //may need to change
				
				layerListObjects.add(conv);
			}
		}

		for (int i = 0; i < layerListObjects.size(); i++) {
			if (layerListObjects.get(i) instanceof ConvolutionalLayer) {

				ConvolutionalLayer conv = (ConvolutionalLayer) layerListObjects.get(i);
				Filters filter = new Filters(conv.numofFilters, conv.filterSize);
				double[][][] array = new double[conv.channelDepth][conv.filterSize][conv.filterSize];

				for (int j = 0; j < array.length; j++) {
					for (int k = 0; k < array[0].length; k++) {
						for (int l = 0; l < array[0][0].length; l++) {
							array[j][k][l] = listOfValues.get(0);
							listOfValues.remove(0);
						}
					}
				}
				filter.threeDFilterArray.add(array);
				filterList.add(filter);

			} else if (layerListObjects.get(i) instanceof HiddenConvolutionalLayer) {
				HiddenConvolutionalLayer hidden = (HiddenConvolutionalLayer) layerListObjects.get(i);
				Filters filter = new Filters(hidden.numofFilters, hidden.filterSize);
				double[][] array = new double[hidden.filterSize][hidden.filterSize];

				for (int j = 0; j < array.length; j++) {
					for (int k = 0; k < array[0].length; k++) {
						array[j][k] = listOfValues.get(0);
						listOfValues.remove(0);
					}
				}
				filter.twoDFilterArray.add(array);
				filterList.add(filter);
			} else if (layerListObjects.get(i) instanceof HiddenLayer) {
				HiddenLayer hidden = (HiddenLayer) layerListObjects.get(i);

				 if (layerListObjects.get(i + 1) instanceof HiddenLayer) {
					HiddenLayer postHidden = (HiddenLayer) layerListObjects.get(i + 1);
					double[][] weight = new double[hidden.layerSize][postHidden.layerSize];
					for (int j = 0; j < weight.length; j++) {
						for (int k = 0; k < weight[0].length; k++) {
							weight[j][k] = listOfValues.get(0);
							listOfValues.remove(0);
						}
					}
					weightList.add(weights.addWeightBiases(weight));
				} else if (layerListObjects.get(i + 1) instanceof OutputLayer) {
					OutputLayer out = (OutputLayer) layerListObjects.get(i + 1);
					double[][] weight = new double[hidden.layerSize][out.layerSize];
					for (int j = 0; j < weight.length; j++) {
						for (int k = 0; k < weight[0].length; k++) {
							weight[j][k] = listOfValues.get(0);
							listOfValues.remove(0);
						}
					}
					weightList.add(weights.addWeightBiases(weight));
				}
			}
		}

	}

	public void acquireTestValues(String testPath) {
		fr = new FileReader(testFilePath + testPath + ".txt");
		fr.initializeFileReader(); // initializes scanner
		fr.readFileIntoList();
		listOfValues = fr.valuesFromFile;

		int numofSets = (int) listOfValues.size() / (int) layerSizes[0];

		inputLayer = new double[numofSets][layerSizes[0]];

		int counter = 0;
		for (int i = 0; i < numofSets; i++) {
			for (int j = 0; j < layerSizes[0]; j++) {
				inputLayer[i][j] = listOfValues.get(counter);
				counter++;
			}
		}
	}

	public void acquireConvTestValues(String testPath) {
		
		try {
			File path = new File(testFilePath + testPath);
			File[] files = path.listFiles();
			
			ConvolutionalLayer conv = (ConvolutionalLayer) layerListObjects.get(0);
			

			if (files[0].getName().endsWith(".txt")) {
				fr = new FileReader(testFilePath + testPath + ".txt");
				imageList = (fr.readImageTextIntoList(conv.channelDepth, conv.imageHeight, conv.imageWidth));

			} else if (files[0].getName().endsWith(".jpg") || files[0].getName().endsWith(".png")) {
				fr = new FileReader(testFilePath + testPath);
				imageList = (fr.readImagesIntoList());
			}

		} catch (Exception e) {
			System.out.println("ERROR");
		}
/*
		fr = new FileReader(testFilePath + testPath + ".txt");
		fr.initializeFileReader(); // initializes scanner
		fr.readFileIntoList();
		listOfValues = fr.valuesFromFile;
*/
		// int numofSets
	}

	public double[][] appendBiasColumn(double[][] layer) {
		double[][] inputsWithBiases = new double[layer.length][layer[0].length + 1];

		for (int i = 0; i < layer.length; i++) {
			for (int j = 0; j < layer[0].length; j++) {
				inputsWithBiases[i][j] = layer[i][j];
			}
		}
		for (int i = 0; i < layer.length; i++) {
			inputsWithBiases[i][layer[0].length] = 1;
		}
		return inputsWithBiases;
	}

}
