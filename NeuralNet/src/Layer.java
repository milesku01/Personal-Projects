import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Layer { // superclass
	double[][] layerValue;
	double[][] preActivatedValue; 
	int layerSize; 
	String activation; 
	
	public Layer(int layerSize) {
		this.layerSize = layerSize; 
	}
	
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
	String fileName = "";
	FileReader fileReader;
	Normalizer normalizer = new Normalizer();
	
	public InputLayer(int numofSets, int numofInput, String fileName) {
		super(numofInput);
		this.numofSets = numofSets; 
		this.numofInput = numofInput;
		this.fileName = fileName; 
	}
	
	public void initializeLayer(InputLayer inputLayer, Targets targets) { //add error handling
		fileReader = new FileReader(fileName);
		inputLayer.setLayerValue(fileReader.readInputIntoArray(numofSets, numofInput)); 
		targets.targetSize = fileReader.determineTargetSize(numofSets, numofInput);
		inputLayer.layerValue = shuffleArray(inputLayer.layerValue); 
		
		targets.determineTargets(inputLayer.layerValue, numofInput); 
		inputLayer.setLayerValue(extractInputs(inputLayer.layerValue));
		inputLayer.setLayerValue(normalizer.normalizeInputs(inputLayer.layerValue)); 
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
		super(numofNeuron);
		this.numofNeuron = numofNeuron;
		this.activation = activation; 
	}
}

class OutputLayer extends Layer {
	int numofOutputNeuron = 0;
	
	public OutputLayer(int numofOutputNeuron, String activation) {
		super(numofOutputNeuron);
		this.numofOutputNeuron = numofOutputNeuron;
		this.activation = activation; 
	}
}