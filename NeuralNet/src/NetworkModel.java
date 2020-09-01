import java.util.ArrayList;
import java.util.List;

public class NetworkModel {
	static int weightListCount = 0;
	static int layerPositionCount = 0;


	static String modelType = ""; 

	public List<Layer> layerList = new ArrayList<Layer>();


	public void buildInputLayer(int numofSets, int numofInputs, int batchSize, String filePath) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "STANDARD"; 
		if (layerList.isEmpty())
			inputLayer.initializeLayer();
		weightListCount++; 
		
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(inputLayer);
	} 
	
	public void buildInputLayerDiagnostic(int numofSets, int numofInputs, int batchSize, String filePath) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "DIAGNOSTIC"; 
		
		if (layerList.isEmpty())
			inputLayer.initializeDiagnosticLayer();
		weightListCount++; 
	
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		
		layerList.add(inputLayer);
	} 
	
	public void buildInputLayerText(int numofSets, int numofInputs, int batchSize, String filePath, String lookup) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		modelType = "STANDARD";
		if (layerList.isEmpty())
			inputLayer.initializeLayerText(lookup);
		weightListCount++; 
		
		inputLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(inputLayer);
	}
	
	public void buildDropoutLayer(int numofNeurons, double probability, String activation) {
		DropoutLayer dropout = new DropoutLayer(numofNeurons, probability, activation);
		weightListCount++; 
		
		dropout.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(dropout);
	}
	
	public void buildHiddenLayer(int numofNeurons, String activation) {
		HiddenLayer hiddenLayer = new HiddenLayer(numofNeurons, activation);
		weightListCount++;
		
		hiddenLayer.layerPosition = layerPositionCount;
		layerPositionCount++; 
		
		layerList.add(hiddenLayer);
	}

	public void buildOutputLayer(int numofNeurons, String activation) {
		OutputLayer outputLayer = new OutputLayer(numofNeurons, activation);
		
		outputLayer.layerPosition = layerPositionCount;
		
		layerList.add(outputLayer);
	} 

}
;