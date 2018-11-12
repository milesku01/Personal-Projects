import java.util.ArrayList;
import java.util.List;

public class NetworkModel {
	public List layerList = new ArrayList();
	public Targets targets = new Targets(); 
	public int batchSize;
	public int numofEpochs;
	
	public NetworkModel(int batchSize, int numofEpochs) {
		this.batchSize = batchSize;
		this.numofEpochs = numofEpochs;
	}
	
	public void buildInputLayer(String filePath, int numofSets, int numofInputs) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, filePath); // no activation
		inputLayer.initializeLayer(inputLayer, targets);
		layerList.add(inputLayer);
	
	}
	public void buildHiddenLayer(int numofNeurons, String activation) {
		HiddenLayer hiddenLayer = new HiddenLayer(numofNeurons, activation);
		layerList.add(hiddenLayer);
	}
	public void buildOutputLayer(int numofNeurons, String activation) {
		OutputLayer outputLayer = new OutputLayer(numofNeurons, activation); 
		layerList.add(outputLayer);
	}
	
}
