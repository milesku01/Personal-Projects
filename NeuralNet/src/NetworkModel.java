import java.util.ArrayList;
import java.util.List;

/**
 * Class NetworkModel is an abstraction for a network model that contains 
 * a list of layer objects that make up the network 
 * 
 * NetworkModel also contains methods to build many different variations of neural networks
 * tailored to any task 
 * 
 * @author kuhnm
 *
 */
public class NetworkModel {
	static int layerPositionCount = 0; //stores where in the model a layer is (0,1,2...)

	String modelType = ""; //a string that defines what kind of model it is ("STANDARD", "DIAGNOSTIC", "EVALUATOR")

	public List<Layer> layerList = new ArrayList<Layer>(); //stores a list of the layers that define the model 

	/**
	 * method buildInputLayer sets up the input layer for the network model in the standard case
	 * An inputLayer is constructed using the constructor in Layer 
	 * 
	 * If the layer hasn't been initialized yet it gets initialized 
	 * 
	 * The layer is assigned its position 
	 * And the layer added to the list 
	 * 
	 * @param numofSets: the number of sets of data (rows) 
	 * @param numofInputs: the number of statistical input per data (columns) 
	 * @param batchSize: the size of the batch
	 * @param filePath: the location of the input data
	 */
	public void buildInputLayer(int numofSets, int numofInputs, int batchSize, String filePath) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "STANDARD"; 
		if (layerList.isEmpty())
			inputLayer.initializeLayer();
		
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(inputLayer);
	} 
	
	/**
	 * method buildInputLayerDiagnostic builds the input layer for a diagnostic type network model used to 
	 * check the legitimacy of the model given any coding changes
	 * 
	 * Set the type of the model to "DIAGNOSTIC" 
	 * 
	 * If the network model hasn't been setup yet the initialize it 
	 * 
	 * Set the position of the layer within the model 
	 * Add the layer to the list 
	 * 
	 * @param numofSets: the number of sets of data (rows) 
	 * @param numofInputs: the number of statistical input per data (columns) 
	 * @param batchSize: the size of the batch
	 * @param filePath: the location of the input data
	 */
	public void buildInputLayerDiagnostic(int numofSets, int numofInputs, int batchSize, String filePath) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "DIAGNOSTIC"; 
		
		if (layerList.isEmpty())
			inputLayer.initializeDiagnosticLayer();
	
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		
		layerList.add(inputLayer);
	} 
	
	/**
	 * method buildInputLayerText builds a inputLayer that uses text as input and then parses that text
	 * 
	 * The type of the model being created is standard and the modelType is set to that
	 * If the model is empty initialize the layer
	 * 
	 * Set the layerPostion of the layer 
	 * Add the layer to the list
	 * 
	 * @param numofSets: the number of sets of data (rows) 
	 * @param numofInputs: the number of statistical input per data (columns) 
	 * @param batchSize: the size of the batch
	 * @param filePath: the location of the input data
	 * @param lookup: the name of the lookup file used to parse the data of the input file 
	 */
	public void buildInputLayerText(int numofSets, int numofInputs, int batchSize, String filePath, String lookup) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "STANDARD";
		if (layerList.isEmpty())
			inputLayer.initializeLayerText(lookup);
		
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(inputLayer);
	}
	
	/**
	 * method buildInputLayerEvaluator builds an inputLayer for the evaluation network which uses a model previously created to 
	 * predict values 
	 * 
	 * Set the model type of the network to evaluator
	 * Add the layer to the list
	 * 
	 * @param numofInputs: the number of inputs per data entry (number of columns) 
	 * @param inputLayerValue: the numerical value of the inputLayer to be evaluated
	 */
	public void buildInputLayerEvaluator(int numofInputs, double[][] inputLayerValue) {
		InputLayer inputLayer = new InputLayer(numofInputs, inputLayerValue); // no activation
		modelType = "EVALUATOR"; 
		
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(inputLayer);
	}
	
	/**
	 * method build drropoutLayer builds a hidden dropoutLayer in the network, a specialized form of dense layer that 
	 * periodically drops neurons so the network has to find new pathways
	 * 
	 * @param numofNeurons: the number of neurons initially active in the hidden layer
	 * @param probability: the probability that at any given time a neuron pathway will dropout
	 * @param activation: the activation used after the layer used to create non linearity in the network 
	 */
	public void buildDropoutLayer(int numofNeurons, double probability, String activation) {
		DropoutLayer dropout = new DropoutLayer(numofNeurons, probability, activation);
		
		dropout.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(dropout);
	}
	
	/**
	 * method buildHiddenLayer builds hidden layers in the network which increases the complexity of the network
	 * many hidden layers can be chained together
	 * 
	 * @param numofNeurons: the number of neurons in the network
	 * @param activation: the activation string of the layer which creates non-linearity 
	 */
	public void buildHiddenLayer(int numofNeurons, String activation) {
		HiddenLayer hiddenLayer = new HiddenLayer(numofNeurons, activation);
		
		hiddenLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(hiddenLayer);
	}

	/**
	 * method buildOutputLayer builds the last layer of the network and holds the value of the output 
	 * can only be created at the end 
	 * 
	 * @param numofNeurons: the number of values at the output, usually one or two but can be infinitely many 
	 * @param activation: activation string at the output, usually softmax or linear
	 */
	public void buildOutputLayer(int numofNeurons, String activation) {
		OutputLayer outputLayer = new OutputLayer(numofNeurons, activation);
		
		outputLayer.layerPosition = layerPositionCount;
		layerPositionCount = 0;
		
		layerList.add(outputLayer);
		
	} 

}
;