import java.util.ArrayList;
import java.util.List;

public class NetworkModel {
	public List<Layer> layerList = new ArrayList<Layer>();
	public Targets targets = new Targets(); 
	
	
	public void buildInputLayer(String filePath, int numofSets, int numofInputs, int batchSize) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
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

	public void buildConvolutionalLayer() {
		ConvolutionalLayer convLayer = new ConvolutionalLayer(); 
		layerList.add(convLayer);
	}
	
	public void buildPoolingLayer() {
		PoolingLayer poolLayer = new PoolingLayer(); 
		layerList.add(poolLayer);
	}
	
	
	
}
