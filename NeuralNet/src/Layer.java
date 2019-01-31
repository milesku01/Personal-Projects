import java.util.ArrayList;
import java.util.Collections;
import java.util.List;



public class Layer { // superclass
	double[][] layerValue;
	double[][] currentBatch;
	double[][] preActivatedValue; 
	double[][] testData; 
	int layerSize;
	static int globalNumofSets;
	String activation; 
	ForwardPropagator fp = new ForwardPropagator(); 
	
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
		inputLayer.layerValue = (normalizer.normalizeInputsZscore(inputLayer.layerValue, targets.targetSize)); 
		
		if(numofSets > 90) {
			trainTestSplit(inputLayer, targets.targetSize); 
			initializeTestData(inputLayer, targets);
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
		List<double[]> list = new ArrayList<double[]>();
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
		target.determineConvolutionalTargets(globalNumofSets, numofOutputNeuron, targetFile);
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
	String folderName;
	String textFile; 
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\";
	double[][][] currentImage; 
	List<double[][][]> imageList; 
	List<double[][][]> trainingImages;
	List<double[][][]> testingImages; 
	FileReader fileReader;
	
	public ConvolutionalLayer(int numofFilters, int filterSize, int strideLength, String padding) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
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
	}
	
	
	public void initializeLayer(ConvolutionalLayer convLayer) { //add error handling
		fileReader = new FileReader(strdFilePath + folderName);
		convLayer.imageList = (fileReader.readImagesIntoList()); 
		convLayer.imageHeight = convLayer.imageList.get(0)[0].length;
		convLayer.imageWidth = convLayer.imageList.get(0)[0][0].length;
		numofSets = convLayer.imageList.size();
		globalNumofSets = numofSets; 
		//shuffleArray(convLayer); 
		
		if(imageList.size() > 90) {
			trainTestSplit(convLayer); 
			imageList.clear();
		} else {
			convLayer.trainingImages = imageList;
		}
		
	}
	
	public void initializeLayerText(ConvolutionalLayer convLayer) {
		fileReader = new FileReader(strdFilePath + textFile + ".txt");
		convLayer.imageList = (fileReader.readImageTextIntoList(channelDepth, imageHeight, imageWidth)); 
		numofSets = convLayer.imageList.size();
		globalNumofSets = numofSets; 
		//shuffleArray(convLayer); 
		if(imageList.size() > 90) {
			trainTestSplit(convLayer); 
			imageList.clear();
		} else {
			convLayer.trainingImages = imageList;
		}
		
	}
	
	
	public void shuffleArray(ConvolutionalLayer conv) {
		Collections.shuffle(conv.imageList);
	}
	
	public void trainTestSplit(ConvolutionalLayer conv) {
		int trainingSize = (int)(.7 * numofSets); 
		int testingSize = numofSets-trainingSize;
		int counter = 0;
		
		conv.trainingImages = new ArrayList<double[][][]>();
		conv.testingImages = new ArrayList<double[][][]>();
		
		System.out.println(numofSets);
		
		for(int i=0; i<trainingSize; i++) {
			conv.trainingImages.add(imageList.get(counter));
			counter++;
		} 
		for(int i=0; i<testingSize; i++) {
			conv.testingImages.add(imageList.get(counter)); 
		}
	}
	
	
}

class PoolingLayer extends Layer {
	int poolSize;
	String poolType; 
	double[][] expandedLayer;
	public PoolingLayer(int poolSize, String poolType) {
		this.poolSize = poolSize;
		this.poolType = poolType;
	}
}


class HiddenConvolutionalLayer extends Layer {
	int filterSize; 
	int numofFilters; 
	int strideLength;
	String padding; 
	double[][] fullyConvolvedDerivative; //change name
	List<double[][]> filterList; 
	
	public HiddenConvolutionalLayer(int numofFilters, int filterSize, int strideLength, String padding) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize; 
		this.strideLength = strideLength; 
		this.padding = padding; 
	}
}

class ReluLayer extends Layer {	
	public ReluLayer() {
		activation = "RELU";
	}
}







