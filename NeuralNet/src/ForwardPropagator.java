import java.util.ArrayList;
import java.util.List;

public class ForwardPropagator {
	int objectTracker = 0;
	static int layerCounter = 0;
	int testObjectTracker = 0;
	static int batchSize = 1;
	static int remainingBatchSize = 0;
	static int hiddenConvolutionalSize = 0;
	double[][] layerValue;
	double[][] currentBatch;
	ForwardPropagator forwardPropObj;
	List<ForwardPropagator> propagationObjects = new ArrayList<ForwardPropagator>();
	List<ForwardPropagator> testPropagationObjects = new ArrayList<ForwardPropagator>();
	static List<Layer> layerList = new ArrayList<Layer>();
	static List<double[][]> weightList = new ArrayList<double[][]>();
	static InputLayer inputLayer;
	static NetworkTrainer nt = new NetworkTrainer();
	static Weights weights;
	Activator activator = new Activator();

	public double[][] propagate(Layer layer, Layer nextLayer) {
		layerValue = propagationObjects.get(objectTracker).propagate(layer, nextLayer);

		if ((layer instanceof ConvolutionalLayer || layer instanceof PoolingLayer || layer instanceof ReluLayer)
				&& (nextLayer instanceof HiddenLayer || nextLayer instanceof OutputLayer)) {
			System.out.println("Called");
			layerValue = flatten(layerValue);
		}

		if (objectTracker == (propagationObjects.size() - 1)) {
			objectTracker = 0;
			layerCounter = 0;
		} else {
			objectTracker++;
		}

		return layerValue;
	}

	public double[][] propagateTest(Layer layer, Layer nextLayer) {
		if (testObjectTracker == 0)
			layerCounter = 0;

		layerValue = testPropagationObjects.get(testObjectTracker).propagate(layer, nextLayer);

		if (testObjectTracker == (testPropagationObjects.size() - 1)) {
			testObjectTracker = 0;
			layerCounter = 0;
		} else {
			testObjectTracker++;
		}

		return layerValue;
	}

	public void constructForwardPropagationObjects(List<Layer> layerList, Weights weights) { // only occur once

		setupConstants(layerList, weights);

		if (layerList.get(0).globalNumofSets < 140) {
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
				hiddenConvolutionalSize++;
			}

			propagationObjects.add(forwardPropObj);

			if (layerList.get(0).globalNumofSets > 140) {
				if (i > 0) {
					testPropagationObjects.add(forwardPropObj);
				}
			}
		}
		System.out.println(propagationObjects.size());
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

	public double[][] flatten(double[][] input) {
		int counter = 0;
		double[][] output = new double[1][input.length * input[0].length];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				output[0][counter] = input[i][j];
				counter++;
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
		layerCounter++;
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue; // it's here for a reason
		nextLayer.layerValue = activate(nextLayer);
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
		System.out.println(layer.layerValue.length + " here " + layer.layerValue[0].length);

		layer.layerValue = appendBiasColumn(layer);
		System.out.println(layerCounter);
		layerValue = nt.matrixMultiplication(layer.layerValue, weightList.get(layerCounter));
		layerCounter++;
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue;
		nextLayer.layerValue = activate(nextLayer);

		System.out.println("OutputLayer " + java.util.Arrays.deepToString(nextLayer.layerValue));

		return nextLayer.layerValue;
	}
}

class TestPropagator extends ForwardPropagator {
	int counter = 0;

	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] testValue;
		if (counter == 0)
			layer.testData = appendBiasColumn(layer);
		counter++;
		testValue = nt.matrixMultiplication(layer.testData, weightList.get(0));
		nextLayer.layerValue = testValue;
		nextLayer.testData = activate(nextLayer);
		return nextLayer.testData;
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

}

class ConvolutionalPropagator extends ForwardPropagator {
	ConvolutionalLayer conv;
	double[][][] image;
	List<double[][][]> filters;

	public double[][] propagate(Layer layer, Layer nextLayer) { // called for every convolutional layer
		conv = (ConvolutionalLayer) layer;
		image = getBatch(conv);
		filters = weights.filterList.get(0).threeDFilterArray;

		System.out.print("Filter ");
		for (int k = 0; k < 1; k++) {
			for (int i = 0; i < 1; i++) {
				for (int j = 0; j < 1; j++) {
					System.out.print(filters.get(0)[k][i][j] + " ");
				}
			}
		}
		System.out.println();

		int biasTerm = 1;
		int tracker = 0;
		int colorChannels = image.length;
		int filterSize = filters.get(0)[0].length;
		int strideLength = conv.strideLength;
		int numofXCycles = (image[0][0].length - filterSize) + 1;
		int numofYCycles = (image[0].length - filterSize) + 1;
		double convOutput = 0;

		double[][] convOutputArraySum = new double[numofYCycles][numofXCycles];
		double[][] convOutputArrayAugmented = new double[numofYCycles * filters.size()][numofXCycles];

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
					convOutputArraySum[cornerPosY / strideLength][cornerPosX / strideLength] = convOutput + biasTerm;
					convOutput = 0;
				}
			}

			for (int i = 0; i < numofYCycles; i++) {
				for (int j = 0; j < numofXCycles; j++) {
					convOutputArrayAugmented[i + tracker * numofYCycles][j] = convOutputArraySum[i][j];
				}
			}

			tracker++;
		}

		nextLayer.layerValue = convOutputArrayAugmented;

		System.out.print("Layer 1 ");
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				System.out.print(convOutputArrayAugmented[i][j] + " ");
			}
		}
		System.out.println();

		return convOutputArrayAugmented;
	}

	int batchCounter = 0;

	private double[][][] getBatch(ConvolutionalLayer conv) {
		double[][][] image = conv.imageList.get(batchCounter);

		if (!hasReachedEndofBatch(conv.imageList.size())) {
			batchCounter++;
		} else {
			batchCounter = 0;
		}

		return image;
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
	double[][] input;
	List<double[][]> filters;

	public double[][] propagate(Layer layer, Layer nextLayer) {
		conv = (HiddenConvolutionalLayer) layer;
		input = layer.layerValue;
		filters = weights.filterList.get(getFilterBatch()).twoDFilterArray;

		int biasTerm = 1;
		int tracker = 0;
		int filterSize = filters.get(0).length;
		int strideLength = conv.strideLength;
		int numofXCycles = (input[0].length - filterSize) + 1;
		int numofYCycles = (input.length - filterSize) + 1;
		double convOutput = 0;

		double[][] convOutputArraySum = new double[numofYCycles][numofXCycles];
		double[][] convOutputArrayAugmented = new double[numofYCycles * filters.size()][numofXCycles];

		for (int h = 0; h < filters.size(); h++) {
			for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
				for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {
					for (int j = 0; j < filterSize; j++) {
						for (int k = 0; k < filterSize; k++) {
							convOutput += input[j + cornerPosY][k + cornerPosX] * filters.get(h)[j][k];
						}
					}
					convOutputArraySum[cornerPosY / strideLength][cornerPosX / strideLength] = convOutput + biasTerm;
					convOutput = 0;
				}
			}

			for (int i = 0; i < numofYCycles; i++) {
				for (int j = 0; j < numofXCycles; j++) {
					convOutputArrayAugmented[i + tracker * numofYCycles][j] = convOutputArraySum[i][j];
				}
			}
			tracker++;
		}

		nextLayer.layerValue = convOutputArrayAugmented;

		System.out.print("Layer 1 ");
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				System.out.print(convOutputArrayAugmented[i][j] + " ");
			}
		}
		System.out.println();

		return convOutputArrayAugmented;
	}

	int batchCounter = 0;

	private int getFilterBatch() {
		if (!hasReachedEndofBatch(hiddenConvolutionalSize)) {
			batchCounter++;
		} else {
			batchCounter = 1;
		}

		return batchCounter;
	}

	private boolean hasReachedEndofBatch(int numofBatches) {
		if (numofBatches == batchCounter) {
			return true;
		} else {
			return false;
		}
	}

}

class PoolingPropagator extends ForwardPropagator {
	public double[][] propagate(Layer layer, Layer nextLayer) {
		PoolingLayer pool = (PoolingLayer) layer;

		int strideLength = pool.poolSize;
		int numofXCycles = pool.layerValue[0].length / strideLength;
		int numofYCycles = pool.layerValue.length / strideLength;

		double[][] input = pool.layerValue;

		System.out.println(input.length + " poolpre " + input[0].length);

		double[][] poolArray = new double[pool.poolSize][pool.poolSize];
		double[][] maxPoolOutput = new double[numofYCycles][numofXCycles];

		for (int cornerPosY = 0; cornerPosY < numofYCycles; cornerPosY += strideLength) {
			for (int cornerPosX = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) {
				for (int i = 0; i < pool.poolSize; i++) {
					for (int j = 0; j < pool.poolSize; j++) {
						poolArray[i][j] = input[i + cornerPosY][j + cornerPosX];
					}
				}
				maxPoolOutput[cornerPosY / strideLength][cornerPosX / strideLength] = getMaxValue(poolArray);
			}
		}

		nextLayer.layerValue = maxPoolOutput;

		System.out.println(maxPoolOutput.length + " pool " + maxPoolOutput[0].length);

		return maxPoolOutput;
	}

	private double getMaxValue(double[][] subSample) {
		double max = subSample[0][0];
		for (int i = 0; i < subSample.length; i++) {
			for (int j = 0; j < subSample[0].length; j++) {
				if (subSample[i][j] > max) {
					max = subSample[i][j];
				}
			}
		}
		// System.out.println(max);
		return max;
	}
}

class ReluPropagator extends ForwardPropagator {
	public double[][] propagate(Layer layer, Layer nextLayer) {
		layer.activation = "RELU";
		nextLayer.preActivatedValue = layer.layerValue; // might be trouble w/ references
		nextLayer.layerValue = activate(layer);

		System.out.print("relu ");
		for (int i = 0; i < 5; i++) {
			for (int j = 0; j < 5; j++) {
				System.out.print(nextLayer.layerValue[i][j] + " ");
			}
		}
		System.out.println();

		return nextLayer.layerValue;
	}

}
