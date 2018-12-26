import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Layer { // superclass
	double[][] layerValue;
	double[][] currentBatch;
	double[][] preActivatedValue; 
	double[][] testData; 
	int layerSize; 
	String activation; 
	ForwardPropagator fp = new ForwardPropagator(); 
	
	public double[][] getLayerValue() {
		return layerValue;
	}
 
	public void setLayerValue(double[][] newValue) {
		layerValue = newValue;
	}
}

class InputLayer extends Layer {
	int numofSets = 0; 
	int numofInput = 0;
	int batchSize = 0;
	String fileName = "";
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\";
	FileReader fileReader;
	Normalizer normalizer = new Normalizer();
	
	public InputLayer(int numofSets, int numofInput, int batchSize, String fileName) {
		layerSize = numofInput;
		this.numofSets = numofSets; 
		this.numofInput = numofInput;
		this.batchSize = batchSize; 
		this.fileName = fileName; 
	}
	
	public void initializeLayer(InputLayer inputLayer, Targets targets) { //add error handling
		fileReader = new FileReader(strdFilePath + fileName + ".txt");
		inputLayer.setLayerValue(fileReader.readInputIntoArray(numofSets, numofInput)); 
		targets.targetSize = fileReader.determineTargetSize(numofSets, numofInput);
		//inputLayer.layerValue = shuffleArray(inputLayer.layerValue); 
		inputLayer.setLayerValue(normalizer.normalizeInputs(inputLayer.layerValue, targets.targetSize)); 
		
		if(numofSets > 140) {
			trainTestSplit(inputLayer, targets.targetSize); 
			initializeTestData(inputLayer, targets);
		}
		targets.determineTargets(inputLayer.layerValue, numofInput); 
		inputLayer.setLayerValue(extractInputs(inputLayer.layerValue));
		inputLayer.setLayerValue(fp.appendBiasColumn(inputLayer));
		
	}
	int trainingSize; 
	
	private void trainTestSplit(InputLayer inputLayer, int targetSize) {
		double[][] trainingData; 
		trainingSize = (int)(.7 * numofSets); 
		int testingSize = numofSets-trainingSize; 
		
		if(numofSets > 140) { //roughly 70% of 140 is 100
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
	
	public OutputLayer(int numofOutputNeuron, String activation) {
		layerSize = numofOutputNeuron;
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation; 
	}
}

class ConvolutionalLayer extends Layer{
	public ConvolutionalLayer() {
		
	}
}

class PoolingLayer extends Layer{
	public PoolingLayer() {
		
	}
}

class TestLayer extends Layer {
	
}



