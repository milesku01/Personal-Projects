import java.util.ArrayList;
import java.util.List;

public class NetworkModel {
	public List<Layer> layerList = new ArrayList<Layer>();
	public Targets targets = new Targets();

	public void buildInputLayer(int numofSets, int numofInputs, int batchSize, String filePath) {
		InputLayer inputLayer = new InputLayer(numofSets, numofInputs, batchSize, filePath); // no activation
		if (layerList.isEmpty())
			inputLayer.initializeLayer(inputLayer, targets);
		layerList.add(inputLayer);

	}

	public void buildHiddenLayer(int numofNeurons, String activation) {
		HiddenLayer hiddenLayer = new HiddenLayer(numofNeurons, activation);
		layerList.add(hiddenLayer);
	}

	public void buildHiddenLayer(String activation) {
		HiddenLayer hiddenLayer = new HiddenLayer(getInferedNumOfNeurons(layerList.size()), activation);
		layerList.add(hiddenLayer);
	}

	private int getInferedNumOfNeurons(int layerListSize) {
		int numofNeuronWidth = 0;
		int numofNeuronHeight = 0;
		ConvolutionalLayer conv = (ConvolutionalLayer) layerList.get(0);

		int imageHeight = conv.imageHeight;
		int imageWidth = conv.imageWidth;
		int numofFilters = conv.numofFilters;
		int filterSize = conv.filterSize;
		int strideLength = conv.strideLength;

		numofNeuronHeight = numofFilters * (((imageHeight - filterSize) / strideLength) + 1);
		numofNeuronWidth = ((imageWidth - filterSize) / strideLength) + 1;
		
		for(int i=1; i<layerListSize; i++ ) {
			if(layerList.get(i) instanceof PoolingLayer) {
				numofNeuronHeight /= ((PoolingLayer)layerList.get(i)).poolSize;
				numofNeuronWidth /= ((PoolingLayer)layerList.get(i)).poolSize;
			}
			if(layerList.get(i) instanceof HiddenConvolutionalLayer) {
				HiddenConvolutionalLayer conv2 = (HiddenConvolutionalLayer) layerList.get(i);
				numofNeuronHeight = numofFilters*((numofNeuronHeight-conv2.filterSize)/strideLength + 1);
				numofNeuronWidth = (numofNeuronWidth-conv2.filterSize)/strideLength + 1; 
			}
			
		}
		return (numofNeuronHeight*numofNeuronWidth);
	}

	public void buildOutputLayer(int numofNeurons, String activation) {
		OutputLayer outputLayer = new OutputLayer(numofNeurons, activation);
		layerList.add(outputLayer);
	} 

	public void buildOutputLayer(int numofNeurons, String activation, String targetFile) {
		OutputLayer outputLayer = new OutputLayer(numofNeurons, activation, targetFile);
		outputLayer.initializeTargets(targets);
		layerList.add(outputLayer);
	}

	public void buildConvolutionalLayer(int numofFilters, int filterSize, int strideLength, int batchSize, String padding, String imageFile) {
		ConvolutionalLayer convLayer = new ConvolutionalLayer(numofFilters, filterSize, strideLength, batchSize, padding, imageFile);
		convLayer.initializeLayer(convLayer);
		layerList.add(convLayer);
	}
	
	public void buildConvolutionalLayer(int height, int width, int channelDepth, int numofFilters,
			int filterSize, int strideLength, int batchSize, String padding, String textFile) {
		ConvolutionalLayer convLayer = new ConvolutionalLayer(height, width, channelDepth, numofFilters, filterSize, strideLength, batchSize, padding, textFile);
		convLayer.initializeLayerText(convLayer);
		layerList.add(convLayer);
	}

	public void buildHiddenConvolutionalLayer(int numofFilters, int filterSize, int strideLength, String padding) {
		HiddenConvolutionalLayer hiddenConvLayer = new HiddenConvolutionalLayer(numofFilters, filterSize, strideLength, padding); 
		layerList.add(hiddenConvLayer);
	}

	public void buildPoolingLayer(int poolSize, String poolType) {
		PoolingLayer poolLayer = new PoolingLayer(poolSize, poolType);
		layerList.add(poolLayer);
	}

	public void buildReluLayer() {
		ReluLayer reluLayer = new ReluLayer();
		layerList.add(reluLayer);
	}


}
