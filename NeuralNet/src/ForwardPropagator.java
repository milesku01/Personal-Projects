import java.util.ArrayList;
import java.util.List;

public class ForwardPropagator {
	int objectTracker = 0;
	static int layerCounter = 0;
	static int filterCounter = 0;
	static double heldDropoutProb = 0.0; 
	int testObjectTracker = 0;
	static int batchSize = 1;
	static int remainingBatchSize = 0;
	double[][] layerValue;
	double[][] currentBatch;
	double[][][] convValue;
	ForwardPropagator forwardPropObj;
	List<ForwardPropagator> propagationObjects;
	List<ForwardPropagator> testPropagationObjects;
	static List<Layer> layerList = new ArrayList<Layer>();
	static List<double[][]> weightList = new ArrayList<double[][]>();
	static InputLayer inputLayer;
	static NetworkTrainer nt = new NetworkTrainer();
	static Weights weights;
	Activator activator = new Activator();

	public double[][] propagate(Layer layer, Layer nextLayer) {
		layerValue = propagationObjects.get(objectTracker).propagate(layer, nextLayer);
		
		

		if (objectTracker == (propagationObjects.size() - 1)) {
			objectTracker = 0;
			layerCounter = 0;
			filterCounter = 0;
		} else {
			objectTracker++;
		}

		return layerValue;
	}

	public double[][][] propagateConv(Layer layer, Layer nextLayer) {
		convValue = propagationObjects.get(objectTracker).propagateConv(layer, nextLayer);

		if ((layer instanceof ConvolutionalLayer || layer instanceof PoolingLayer || layer instanceof ReluLayer
				|| layer instanceof HiddenConvolutionalLayer)
				&& (nextLayer instanceof HiddenLayer || nextLayer instanceof DropoutLayer || nextLayer instanceof OutputLayer)) {
			nextLayer.layerValue = flatten(convValue);
		}

		if (objectTracker == (propagationObjects.size() - 1)) {
			objectTracker = 0;
			layerCounter = 0;
			filterCounter = 0;
		} else {
			objectTracker++;
		}

		return convValue;

	}

	public double[][] propagateTest(Layer layer, Layer nextLayer) {
		if (testObjectTracker == 0) {
			layerCounter = 0;
			filterCounter = 0;
		}

		if(layer instanceof DropoutLayer) {
			((DropoutLayer)layer).dropoutProbability = 0.0; 
		}
		
		layerValue = testPropagationObjects.get(testObjectTracker).propagate(layer, nextLayer);

		if(layer instanceof DropoutLayer) {
			((DropoutLayer)layer).dropoutProbability = heldDropoutProb; //clumsily written but should work
		}
	

		if (testObjectTracker == (testPropagationObjects.size() - 1)) {
			testObjectTracker = 0;
			layerCounter = 0;
			filterCounter = 0;
		} else {
			testObjectTracker++;
		}

		return layerValue;
	}
	
	public double[][][] propagateConvTest(Layer layer, Layer nextLayer) {
		if (testObjectTracker == 0) {
			layerCounter = 0;
			filterCounter = 0;
		}
		
		convValue = testPropagationObjects.get(testObjectTracker).propagateConv(layer, nextLayer);

		if ((layer instanceof ConvolutionalLayer || layer instanceof PoolingLayer || layer instanceof ReluLayer
				|| layer instanceof HiddenConvolutionalLayer)
				&& (nextLayer instanceof HiddenLayer || nextLayer instanceof DropoutLayer || nextLayer instanceof OutputLayer)) {
			nextLayer.layerValue = flatten(convValue);
		}
		
		if (testObjectTracker == (testPropagationObjects.size() - 1)) {
			testObjectTracker = 0;
			layerCounter = 0;
			filterCounter = 0;
		} else {
			testObjectTracker++;
		}
		
		return convValue; 
	}

	public void constructForwardPropagationObjects(List<Layer> layerList, Weights weights) { // only occur once

		setupConstants(layerList, weights);

		if (layerList.get(0).globalNumofSets > 90) {
			forwardPropObj = new TestPropagator();
			testPropagationObjects.add(forwardPropObj);
		}

		for (int i = 0; i < layerList.size() - 1; i++) { // minus one because returns last value

			if (layerList.get(i) instanceof HiddenLayer) {
				forwardPropObj = new DensePropagator();
			} else if (layerList.get(i) instanceof InputLayer) {
				forwardPropObj = new InputLayerPropagator();
			} else if (layerList.get(i) instanceof ConvolutionalLayer) {
				forwardPropObj = new ConvolutionalPropagator();
			} else if (layerList.get(i) instanceof PoolingLayer) {
				forwardPropObj = new PoolingPropagator();
			} else if (layerList.get(i) instanceof ReluLayer) {
				forwardPropObj = new ReluPropagator();
			} else if (layerList.get(i) instanceof HiddenConvolutionalLayer) {
				forwardPropObj = new HiddenConvolutionalPropagator();
			} else if (layerList.get(i) instanceof DropoutLayer) {
				forwardPropObj = new DropoutPropagator(); 
			}

			propagationObjects.add(forwardPropObj);

			if (layerList.get(0).globalNumofSets > 90) {
				if (i > 0) {
					testPropagationObjects.add(forwardPropObj);
				}
			}
		}
	}

	private void setupConstants(List<Layer> layerList, Weights weights) {
		ForwardPropagator.layerList = layerList;
		ForwardPropagator.weights = weights;
		if (layerList.get(0) instanceof InputLayer) {
			inputLayer = (InputLayer) layerList.get(0);
			batchSize = inputLayer.batchSize;
			remainingBatchSize = inputLayer.remainingBatchSize;
		}
		ForwardPropagator.weightList = weights.weightList;
		propagationObjects = new ArrayList<ForwardPropagator>(layerList.size());
		testPropagationObjects = new ArrayList<ForwardPropagator>(layerList.size());

	}

	public double[][] appendBiasColumn(Layer layer) {
		double[][] layerValue = copyArray(layer.layerValue);
		double[][] inputsWithBiases = new double[layerValue.length][layerValue[0].length + 1];

		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				inputsWithBiases[i][j] = layerValue[i][j];
			}
		}
		for (int i = 0; i < layerValue.length; i++) {
			inputsWithBiases[i][layerValue[0].length] = 1;
		}
		return inputsWithBiases;
	}

	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer);
		return activatedValue;
	}

	public double[][] copyArray(double[][] input) {
		double[][] copy = new double[input.length][input[0].length];
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				copy[i][j] = input[i][j];
			}
		}
		return copy;
	}

	public double[][][] copyThreeDArray(double[][][] input) {
		double[][][] copy = new double[input.length][input[0].length][input[0][0].length];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				for (int k = 0; k < input[0][0].length; k++) {
					copy[i][j][k] = input[i][j][k];
				}
			}
		}
		return copy;
	}

	public double[][] flatten(double[][][] input) {
		int counter = 0;
		double[][] output = new double[1][input.length * input[0].length * input[0][0].length];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				for (int k = 0; k < input[0][0].length; k++) {
					output[0][counter] = input[i][j][k];
					counter++;
				}
			}
		}
		return output;
	}

}

class InputLayerPropagator extends ForwardPropagator {

	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] layerValue;
		currentBatch = getBatch(layer);
		layer.currentBatch = currentBatch;
		layerValue = nt.matrixMultiplication(currentBatch, weightList.get(layerCounter));
		
//		System.out.println(java.util.Arrays.deepToString(weightList.get(layerCounter))); 
		
		layerCounter++;
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue; // it's here for a reason
		activate(nextLayer);
		//nextLayer.layerValue = activate(nextLayer);
		return nextLayer.layerValue;
	}

	int batchTracker = 0;
	int batchCounter = 0;

	public double[][] getBatch(Layer layer) {
		double[][] batch;

		if (remainingBatchSize == 0) {
			remainingBatchSize = batchSize;
		}

		if (!hasReachedEndofBatch()) {
			batch = new double[batchSize][layer.layerSize + 1];
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < layer.layerSize + 1; j++) { // added one for the bias column
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++;
			}
			batchCounter++;

		} else {
			batch = new double[remainingBatchSize][layer.layerSize + 1];
			for (int i = 0; i < remainingBatchSize; i++) {
				for (int j = 0; j < layer.layerSize + 1; j++) {
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++;
			}
			batchTracker = 0;
			batchCounter = 0;
		}
		return batch;
	}

	public boolean hasReachedEndofBatch() {
		if (nt.numofBatches - 1 == batchCounter) {
			return true;
		} else {
			return false;
		}
	}

}

class DensePropagator extends ForwardPropagator {

	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] layerValue;

		layer.layerValue = appendBiasColumn(layer);
		layerValue = nt.matrixMultiplication(layer.layerValue, weightList.get(layerCounter));
		
	//	System.out.println(java.util.Arrays.deepToString(weightList.get(layerCounter))); 
		
		layerCounter++;
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue;
		nextLayer.layerValue = activate(nextLayer);
		
	//	System.out.print("OutputLayer ");
	//	nt.printArray(nextLayer.layerValue);

		return nextLayer.layerValue;
	}
}

class DropoutPropagator extends ForwardPropagator {
	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] layerValue; 
		
		heldDropoutProb = ((DropoutLayer)layer).dropoutProbability;
		
		layer.layerValue = dropOut(layer);
		layer.layerValue = appendBiasColumn(layer);
		layerValue = nt.matrixMultiplication(layer.layerValue, weightList.get(layerCounter));
		layerCounter++;
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue;
		nextLayer.layerValue = activate(nextLayer);
	
		return nextLayer.layerValue;
	}
	
	private double[][] dropOut(Layer layer) {
		double[][] layerValue = copyArray(layer.layerValue);
		double probability = ((DropoutLayer)layer).dropoutProbability; 
		
		for(int i=0; i<layerValue[0].length; i++) {
			if(Math.random() <= probability) {
				for(int j=0; j < layerValue.length; j++) {
					layerValue[j][i] = 0;
				}
			}
		}
		return layerValue;
	}
}

class TestPropagator extends ForwardPropagator {
	static int counter = 0;

	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] testValue;
		
		if (layer instanceof InputLayer) {
			if (counter == 0) {
				layer.testData = appendBiasColumn(layer);
				counter++; 
			}
			
			testValue = nt.matrixMultiplication(layer.testData, weightList.get(layerCounter));
			layerCounter++; 
			nextLayer.layerValue = testValue;
			nextLayer.testData = activate(nextLayer);

		} 
		return nextLayer.testData; 
	}

	int batchCounter = 0;

	private void getBatch(ConvolutionalLayer conv) {
		conv.currentImage = conv.testingImages.get(batchCounter);

		if (!hasReachedEndofBatch(conv.testingImages.size())) {
			batchCounter++;
		} else {
			batchCounter = 0;
		}

	}

	private boolean hasReachedEndofBatch(int numofBatches) {
		if (numofBatches - 1 == batchCounter) {
			return true;
		} else {
			return false;
		}
	}

	public double[][] appendBiasColumn(Layer layer) {
		double[][] layerValue = copyArray(layer.testData);
		double[][] inputsWithBiases = new double[layerValue.length][layerValue[0].length + 1];

		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				inputsWithBiases[i][j] = layerValue[i][j];
			}
		}
		for (int i = 0; i < layerValue.length; i++) {
			inputsWithBiases[i][layerValue[0].length] = 1;
		}
		return inputsWithBiases;
	}

	
	public double[][][] propagateConv(Layer layer, Layer nextLayer) {
		
		ConvolutionalLayer conv = (ConvolutionalLayer) layer;
		getBatch(conv);
		double[][][] image = conv.currentImage;

		List<double[][][]> filters = weights.filterList.get(filterCounter).threeDFilterArray;

		int biasTerm = 1;
		int colorChannels = image.length;
		int filterSize = filters.get(0)[0].length;
		int strideLength = conv.strideLength;
		int numofXCycles = (image[0][0].length - filterSize) / strideLength + 1;
		int numofYCycles = (image[0].length - filterSize) / strideLength + 1;
		double convOutput = 0;

		double[][][] convOutputArraySum = new double[filters.size()][numofYCycles][numofXCycles];

		for (int h = 0; h < filters.size(); h++) {

			for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
				for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {

					for (int i = 0; i < colorChannels; i++) {
						for (int j = 0; j < filterSize; j++) {
							for (int k = 0; k < filterSize; k++) {
								convOutput += image[i][j + cornerPosY][k + cornerPosX] * filters.get(h)[i][j][k];
							}
						}
					}
					convOutputArraySum[h][cornerPosY / strideLength][cornerPosX / strideLength] = convOutput + biasTerm;
					convOutput = 0;
				}
			}
		}

		filterCounter++;

		nextLayer.preActivatedConvValue = convOutputArraySum; 
		nextLayer.testConvData = convOutputArraySum;
	
		return nextLayer.testConvData;
	}
	
	
}

class ConvolutionalPropagator extends ForwardPropagator {
	ConvolutionalLayer conv;
	double[][][] image;
	List<double[][][]> filters;

	public double[][][] propagateConv(Layer layer, Layer nextLayer) {

		conv = (ConvolutionalLayer) layer;
		getBatch(conv);
		image = conv.currentImage;

		filters = weights.filterList.get(filterCounter).threeDFilterArray;

		int biasTerm = 1;
		int colorChannels = image.length;
		int filterSize = filters.get(0)[0].length;
		int strideLength = conv.strideLength;
		int numofXCycles = (image[0][0].length - filterSize) / strideLength + 1;
		int numofYCycles = (image[0].length - filterSize) / strideLength + 1;
		double convOutput = 0;

		double[][][] convOutputArraySum = new double[filters.size()][numofYCycles][numofXCycles];

		for (int h = 0; h < filters.size(); h++) {

			for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
				for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {

					for (int i = 0; i < colorChannels; i++) {
						for (int j = 0; j < filterSize; j++) {
							for (int k = 0; k < filterSize; k++) {
								convOutput += image[i][j + cornerPosY][k + cornerPosX] * filters.get(h)[i][j][k];
							}
						}
					}
					convOutputArraySum[h][cornerPosY / strideLength][cornerPosX / strideLength] = convOutput + biasTerm;
					convOutput = 0;
				}
			}
		}

		filterCounter++;

		nextLayer.preActivatedConvValue = convOutputArraySum;
		nextLayer.convValue = convOutputArraySum;

		return convOutputArraySum;
	}

	int batchCounter = 0;

	private void getBatch(ConvolutionalLayer conv) {
		conv.currentImage = conv.trainingImages.get(batchCounter);

		if (!hasReachedEndofBatch(conv.trainingImages.size())) {
			batchCounter++;
		} else {
			batchCounter = 0;
		}

	}

	private boolean hasReachedEndofBatch(int numofBatches) {
		if (numofBatches - 1 == batchCounter) {
			return true;
		} else {
			return false;
		}
	}

}

class HiddenConvolutionalPropagator extends ForwardPropagator {
	HiddenConvolutionalLayer conv;
	double[][][] input;
	List<double[][][]> filters;

	public double[][][] propagateConv(Layer layer, Layer nextLayer) {
		conv = (HiddenConvolutionalLayer) layer;
		input = layer.convValue;

		filters = weights.filterList.get(filterCounter).threeDFilterArray;

		int biasTerm = 1;
		int filterSize = filters.get(0)[0].length;
		int strideLength = conv.strideLength;
		int numofXCycles = (input[0][0].length - filterSize) / strideLength + 1;
		int numofYCycles = (input[0].length - filterSize) / strideLength + 1;
		double convOutput = 0;

		double[][][] convOutputArraySum = new double[filters.size()][numofYCycles][numofXCycles];

		for (int h = 0; h < filters.size(); h++) {
			for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
				for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {
					for (int l = 0; l < input.length; l++) {
						for (int j = 0; j < filterSize; j++) {
							for (int k = 0; k < filterSize; k++) {
								convOutput += input[l][j + cornerPosY][k + cornerPosX] * filters.get(h)[l][j][k];
							}
						}
					}
					convOutputArraySum[h][cornerPosY / strideLength][cornerPosX / strideLength] = convOutput + biasTerm;
					convOutput = 0;
				}
			}

		}

		filterCounter++;

		nextLayer.preActivatedConvValue = convOutputArraySum; 
		nextLayer.convValue = convOutputArraySum;

		return convOutputArraySum;
	}

}

class PoolingPropagator extends ForwardPropagator {
	public double[][][] propagateConv(Layer layer, Layer nextLayer) {
		PoolingLayer pool = (PoolingLayer) layer;

		layer.preActivatedConvValue = pool.convValue;

		int strideLength = pool.poolSize;
		int numofXCycles = pool.convValue[0][0].length / strideLength;
		int numofYCycles = pool.convValue[0].length / strideLength;

		double[][][] input = pool.convValue;

		double[][][] poolArray = new double[pool.convValue.length][pool.poolSize][pool.poolSize];
		double[][][] maxPoolOutput = new double[pool.convValue.length][numofYCycles][numofXCycles];

		pool.expandedLayer = fullExpandedMaxArray(pool);

		
		for (int k=0; k < pool.convValue.length; k++) {
		
			for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
				for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {
				
				
					for (int i = 0; i < pool.poolSize; i++) {
						for (int j = 0; j < pool.poolSize; j++) {
							poolArray[k][i][j] = input[k][i + cornerPosY][j + cornerPosX];
						}
					}
				
					maxPoolOutput[k][cornerPosY / strideLength][cornerPosX / strideLength] = getMaxValue(poolArray);
				}
			}
		}	

		nextLayer.preActivatedConvValue = maxPoolOutput;
		nextLayer.convValue = maxPoolOutput;

		return maxPoolOutput;
	}

	private double getMaxValue(double[][][] subSample) {
		double max = subSample[0][0][0];
		for (int i = 0; i < subSample.length; i++) {
			for (int j = 0; j < subSample[0].length; j++) {
				for(int k=0; k < subSample[0][0].length; k++) {
					if (subSample[i][j][k] > max) {
						max = subSample[i][j][k];
					}
				}
			}
		}

		return max;
	}

	private double[][][] fullExpandedMaxArray(PoolingLayer pool) {
		double[][][] input = copyThreeDArray(pool.convValue);
		double max;
		int maxX;
		int maxY;

		for (int i = 0; i < pool.convValue.length; i++) {
			for (int j = 0; j < pool.convValue[0].length; j += pool.poolSize) {
				for (int m = 0; m < pool.convValue[0][0].length; m += pool.poolSize) {
					max = 0;
					maxX = 0;
					maxY = 0;

					for (int k = 0; k < pool.poolSize; k++) {
						for (int l = 0; l < pool.poolSize; l++) {
							if (input[i][j + k][m + l] > max) {
								max = input[i][j + k][m + l];
								maxX = j + k;
								maxY = m + l;
							}
						}
					}

					for (int k = 0; k < pool.poolSize; k++) {
						for (int l = 0; l < pool.poolSize; l++) {
							if ((j + k != maxX) || (m + l != maxY)) {
								input[i][j + k][m + l] = 0;
							} else if ((j + k == maxX) || (m + l == maxY)) {
								input[i][j + k][m + l] = 1;
							}
						}
					}

				}
			}
		}

		return input;
	}

}

class ReluPropagator extends ForwardPropagator {
	
	public double[][][] propagateConv(Layer layer, Layer nextLayer) {
		layer.activation = "RELU";
		nextLayer.preActivatedConvValue = layer.convValue; // might be trouble w/ references
		nextLayer.convValue = activateRelu(layer);
		return nextLayer.convValue;
	}
	
	private double[][][] activateRelu(Layer layer) {
		double[][][] output = copyThreeDArray(layer.preActivatedConvValue); 
		
		for(int i=0; i<output.length; i++) {
			for(int j=0; j<output[0].length; j++) {
				for(int k=0; k<output[0][0].length; k++) {
					if(output[i][j][k] < 0) {
						output[i][j][k] = 0;
					}
				}
			}
		}
		return output; 
	}

}
