import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;



public class Layer { // superclass
	double[][] layerValue;
	double[][] currentBatch;
	double[][] preActivatedValue; 
	double[][] testData; 
	double[][][] testConvData;
	double[][][] convValue;
	double[][][] preActivatedConvValue; 
	int layerSize;
	static int globalNumofSets;
	final static long seed = (long)(1 + (new Random().nextFloat() * (10000 - 1)));
	String activation; 
	ForwardPropagator fp = new ForwardPropagator(); 
	
	public int parseObjectTypeIntoInt(Layer layer) {
		int output = 0;
		if(layer instanceof InputLayer) {
			output = 0;
		} else if(layer instanceof HiddenLayer) {
			output = 1;
		} else if(layer instanceof ConvolutionalLayer) {
			ConvolutionalLayer conv = (ConvolutionalLayer) layer; 
			if(conv.type == "TEXT") {
				output = 2; 
			} else if(conv.type == "IMAGE") {
				output = 7; 
			}
		} else if(layer instanceof HiddenConvolutionalLayer) {
			output = 3;
		} else if(layer instanceof PoolingLayer) {
			output = 4;
		} else if(layer instanceof ReluLayer) {
			output = 5;
		} else if(layer instanceof OutputLayer) {
			output = 6; 
		}
		return output; 
	}
	
}

class InputLayer extends Layer {
	int numofSets = 0; 
	int numofInput = 0;
	int batchSize = 0;
	int remainingBatchSize = 0; 
	String fileName = "";
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\";
	FileReader fileReader;
	Normalizer normalizer = new Normalizer();
	
	public InputLayer(int numofSets, int numofInput, int batchSize, String fileName) {
		layerSize = numofInput;
		this.numofSets = numofSets; 
		globalNumofSets = numofSets;
		this.numofInput = numofInput;
		this.batchSize = batchSize; 
		this.fileName = fileName; 
		remainingBatchSize = (numofSets % batchSize);
	}
	
	public void initializeLayer(InputLayer inputLayer, Targets targets) { //add error handling
		fileReader = new FileReader(strdFilePath + fileName + ".txt");
		inputLayer.layerValue = (fileReader.readInputIntoArray(numofSets, numofInput)); 
		targets.targetSize = fileReader.determineTargetSize(numofSets, numofInput);
		inputLayer.layerValue = shuffleArray(inputLayer.layerValue); 
		//inputLayer.layerValue = (normalizer.normalizeInputsZscore(inputLayer.layerValue, targets.targetSize)); 
		
		if(numofSets > 90) {
			trainTestSplit(inputLayer, targets.targetSize); 
			inputLayer.layerValue = normalizer.normalizeInputsZscore(inputLayer.layerValue, targets.targetSize);
			initializeTestData(inputLayer, targets);
			inputLayer.testData = normalizer.normalizeInputs(inputLayer.testData, normalizer.meanArray, normalizer.strdDev);
		} else {
			inputLayer.layerValue = normalizer.normalizeInputsZscore(inputLayer.layerValue, targets.targetSize);
		}
		
		targets.determineTargets(inputLayer.layerValue, numofInput); 
		inputLayer.layerValue = (extractInputs(inputLayer.layerValue));
		inputLayer.layerValue = (fp.appendBiasColumn(inputLayer));
		
	}
	
	int trainingSize; 
	
	private void trainTestSplit(InputLayer inputLayer, int targetSize) {
		double[][] trainingData; 
		trainingSize = (int)(.7 * numofSets); 
		remainingBatchSize = (trainingSize % batchSize);
		
		int testingSize = numofSets-trainingSize; 
		
		if(numofSets > 90) { //roughly 70% of 140 is 100
			trainingData = new double[trainingSize][numofInput + targetSize]; 
			testData = new double[testingSize][numofInput + targetSize]; 
			
			for(int i=0; i<trainingSize; i++) {
				for(int j=0; j<numofInput + targetSize; j++) {
					trainingData[i][j] = inputLayer.layerValue[i][j];
				}
			}
			for(int i=trainingSize; i<numofSets; i++) {
				for(int j=0; j<numofInput + targetSize; j++) {
					testData[i-trainingSize][j] = inputLayer.layerValue[i][j]; 
				}
			}
			
			inputLayer.layerValue = trainingData; 
		}
	}
	
	private void initializeTestData(InputLayer inputLayer, Targets target) {
		target.determineTestTargets(inputLayer.testData, numofInput, trainingSize);
		inputLayer.testData= extractInputs(inputLayer.testData); 
	}
	
	private double[][] shuffleArray(double[][] inputLayer) {
		List<double[]> list = new ArrayList<double[]>(inputLayer.length);
		double[] array = null; 
		
		for(int i=0; i<inputLayer.length; i++) {
			array = new double[inputLayer[0].length]; 
			for(int j=0; j<inputLayer[0].length; j++) {
				array[j] = inputLayer[i][j];  
			}
			list.add(array);
		}
		
		Collections.shuffle(list);
		
		for(int i=0; i<inputLayer.length; i++) {
			for(int j=0; j<inputLayer[0].length; j++) {
				inputLayer[i][j] = list.get(i)[j];
			}
		}
		return inputLayer; 
	}
	
	public double[][] extractInputs(double[][] inputs) {
		int targetSize = fileReader.determineTargetSize(numofSets, numofInput);
		double[][] result  = new double[inputs.length][inputs[0].length - targetSize]; 
		
		for(int i=0; i < inputs.length; i++) {
			for(int j=0; j < inputs[0].length - targetSize; j++) {
				result[i][j] = inputs[i][j]; 
			}
		}
		return result; 
	}

} // end of class inputlayer

class HiddenLayer extends Layer {
	int numofNeuron = 0;
	public HiddenLayer(int numofNeuron, String activation) {
		layerSize = numofNeuron;
		this.numofNeuron = numofNeuron;
		this.activation = activation; 
	}
	
}

class OutputLayer extends Layer {
	int numofOutputNeuron = 0;
	String targetFile; 

	public OutputLayer(int numofOutputNeuron, String activation) {
		layerSize = numofOutputNeuron;
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation; 
	}
	
	public OutputLayer(int numofOutputNeuron, String activation, String targetFile) {
		layerSize = numofOutputNeuron; 
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation;
		this.targetFile = targetFile;
	}
	
	public void initializeTargets(Targets target) { //only called for covnet until cleaned
		target.targetSize = numofOutputNeuron;
		System.out.println(seed);
		target.determineConvolutionalTargets(globalNumofSets, numofOutputNeuron, seed, targetFile);
	}
	
}

class DropoutLayer extends Layer {
	int numofNeuron = 0;
	double dropoutProbability = 0;
	public DropoutLayer(int numofNeuron, double dropoutProbability, String activation) {
		layerSize = numofNeuron;
		this.numofNeuron = numofNeuron;
		this.activation = activation;
		this.dropoutProbability = dropoutProbability; 
	}
}

class ConvolutionalLayer extends Layer {
	int filterSize; 
	int numofFilters; 
	int strideLength;
	int numofSets; 
	int imageHeight;
	int imageWidth;
	int batchSize;
	int channelDepth; 
	String padding; 
	String type; //text or images
	String folderName;
	String textFile; 
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\";
	double[][][] currentImage; 
	List<double[][][]> imageList; 
	List<double[][][]> trainingImages;
	List<double[][][]> testingImages; 
	FileReader fileReader;
	Normalizer normalizer = new Normalizer(); 
	
	public ConvolutionalLayer(int numofFilters, int filterSize, int strideLength, String padding) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
		batchSize = 1; //if need
		this.padding = padding; 
	}

	public ConvolutionalLayer(int numofFilters, int filterSize,	int strideLength, int batchSize, String padding,
			String folderName) { 
		this.numofFilters = numofFilters;
		this.folderName = folderName; 
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
		this.batchSize = batchSize; 
		this.padding = padding;	
		type = "IMAGE";
	}
	
	public ConvolutionalLayer(int height, int width, int channelDepth, int numofFilters, int filterSize, int strideLength,
			int batchSize, String padding, String textFile) {
		imageHeight = height;
		imageWidth = width;
		this.channelDepth = channelDepth;
		this.numofFilters = numofFilters;
		this.textFile = textFile; 
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
		this.batchSize = batchSize; 
		this.padding = padding;	
		type = "TEXT";
	}
	
	public ConvolutionalLayer(int height, int width, int channelDepth, int numofFilters, int filterSize, int strideLength, String padding) {
		imageHeight = height;
		imageWidth = width;
		this.channelDepth = channelDepth;
		this.numofFilters = numofFilters;
		this.filterSize = filterSize; 
		this.strideLength = strideLength;
		this.padding = padding;	
	}
	
	
	public void initializeLayer(ConvolutionalLayer convLayer) { //add error handling
		fileReader = new FileReader(strdFilePath + folderName);
		convLayer.imageList = (fileReader.readImagesIntoList()); 
		convLayer.channelDepth = convLayer.imageList.get(0).length;
		convLayer.imageHeight = convLayer.imageList.get(0)[0].length;
		convLayer.imageWidth = convLayer.imageList.get(0)[0][0].length;
		numofSets = convLayer.imageList.size();
		globalNumofSets = numofSets; 
		shuffleArray(convLayer); 
		
	//	imageList = normalizer.normalizeImagesZscore(imageList);
	
		if(imageList.size() > 90) {
			trainTestSplit(convLayer); 
			convLayer.trainingImages = normalizer.normalizeImagesZscore(convLayer.trainingImages); 
			convLayer.testingImages = normalizer.normalizeImagesZscore(convLayer.testingImages, normalizer.imageMean, normalizer.imageStrdDev);
			imageList.clear();
		} else {
			convLayer.trainingImages = imageList;
			convLayer.trainingImages = normalizer.normalizeImagesZscore(convLayer.trainingImages); 
		}
		
		
		
		
		
	}
	
	public void initializeLayerText(ConvolutionalLayer convLayer) {
		fileReader = new FileReader(strdFilePath + textFile + ".txt");
		convLayer.channelDepth =  channelDepth; 
		convLayer.imageList = (fileReader.readImageTextIntoList(channelDepth, imageHeight, imageWidth)); 
		numofSets = convLayer.imageList.size();
		globalNumofSets = numofSets; 
		shuffleArray(convLayer); 
		
		imageList = normalizer.normalizeImagesZscore(imageList);
		
		
		
		if(imageList.size() > 90) {
			trainTestSplit(convLayer); 
			convLayer.trainingImages = normalizer.normalizeImagesZscore(convLayer.trainingImages); 
			convLayer.testingImages = normalizer.normalizeImagesZscore(convLayer.testingImages, normalizer.imageMean, normalizer.imageStrdDev);
			imageList.clear();
		} else {
			convLayer.trainingImages = imageList;
			convLayer.trainingImages = normalizer.normalizeImagesZscore(convLayer.trainingImages); 
		}
		
	}
	
	
	public void shuffleArray(ConvolutionalLayer conv) {
		Collections.shuffle(conv.imageList, new Random(seed));
	}
	
	public void trainTestSplit(ConvolutionalLayer conv) {
		int trainingSize = (int)(.7 * numofSets); 
		int testingSize = numofSets-trainingSize;
		int counter = 0;
		
		conv.trainingImages = new ArrayList<double[][][]>(trainingSize);
		conv.testingImages = new ArrayList<double[][][]>(testingSize);
		
		
		for(int i=0; i<trainingSize; i++) {
			conv.trainingImages.add(imageList.get(counter));
			counter++;
		} 
		for(int i=0; i < testingSize; i++) {
			conv.testingImages.add(imageList.get(counter)); 
			counter++;
		}
		
	}
	
	
}

class PoolingLayer extends Layer {
	int poolSize;
	int channelDepth;
	String poolType; 
	double[][][] expandedLayer;
	public PoolingLayer(int poolSize, String poolType) {
		this.poolSize = poolSize;
		this.poolType = poolType;
	}
}


class HiddenConvolutionalLayer extends Layer {
	int filterSize; 
	int channelDepth; 
	int numofFilters; 
	int strideLength;
	String padding; 
	double[][][] fullyConvolvedDerivative; //change name
	List<double[][][]> filterList; 
	
	public HiddenConvolutionalLayer(int numofFilters, int filterSize, int strideLength, String padding) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
		this.padding = padding; 
	}
}

class ReluLayer extends Layer {	
	int channelDepth;
	public ReluLayer() {
		activation = "RELU";
	}
}







