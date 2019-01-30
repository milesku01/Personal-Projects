import java.util.ArrayList;
import java.util.List;

public class BackPropagator {
	final double regularize = 0.001;
	int objectTracker = 0;
	static int layerCounter;
	static int batchSize;
	static int remainingBatchSize;
	static double[][] gradient;
	BackPropagator backPropObj;
	List<BackPropagator> propagationObjects = new ArrayList<BackPropagator>();
	static List<Layer> layerList;
	static List<double[][]> weightList;
	static double[][] previousPartialGradient;
	static NetworkTrainer nt = new NetworkTrainer();
	static InputLayer inputLayer;
	static Targets targets;
	static Gradients gradients;
	Activator activator = new Activator();

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		if ((layer instanceof HiddenLayer && nextLayer instanceof HiddenLayer)
				|| (layer instanceof HiddenLayer && nextLayer instanceof InputLayer) || layer instanceof OutputLayer
				|| (layer instanceof ReluLayer && nextLayer instanceof ConvolutionalLayer)) {
			gradients = propagationObjects.get(objectTracker).computeGradients(layer, nextLayer);
		}

		else if ((layer instanceof ReluLayer && (nextLayer instanceof ConvolutionalLayer) == false)
				|| (layer instanceof PoolingLayer) || (layer instanceof HiddenLayer
						&& (nextLayer instanceof PoolingLayer || nextLayer instanceof ReluLayer))) {
			propagationObjects.get(objectTracker).computeHiddenGradients(layer, nextLayer);
		}

		if (objectTracker == (propagationObjects.size() - 1)) {
			objectTracker = 0;
			layerCounter = weightList.size(); // once propagated over output, will decrease by one
		} else {
			objectTracker++;
			layerCounter--;
		}

		return gradients;
	}

	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		// return gradients;
	}

	public void constructBackwardPropagationObjects(NetworkModel model, Weights weights) { // only occur once

		setupConstants(model, weights);

		for (int i = layerList.size() - 1; i >= 0; i--) { // minus one because returns last value
			if (layerList.get(i) instanceof HiddenLayer) {
				backPropObj = new DenseBackPropagator();
				propagationObjects.add(backPropObj);
			} else if (layerList.get(i) instanceof OutputLayer) {
				backPropObj = new OutputBackPropagator();
				propagationObjects.add(backPropObj);
			} else if (layerList.get(i) instanceof HiddenConvolutionalLayer) {
				backPropObj = new HiddenConvolutionalBackPropagator();
				propagationObjects.add(backPropObj);
			} else if (layerList.get(i) instanceof PoolingLayer) {
				backPropObj = new PoolingBackPropagator();
				propagationObjects.add(backPropObj);
			} else if (layerList.get(i) instanceof ReluLayer) {
				backPropObj = new ReluBackPropagator();
				propagationObjects.add(backPropObj);
			}
		}

	}

	private void setupConstants(NetworkModel model, Weights weights) {
		BackPropagator.layerList = model.layerList;
		targets = model.targets;

		if (model.layerList.get(0) instanceof InputLayer) {
			inputLayer = (InputLayer) layerList.get(0);
			batchSize = inputLayer.batchSize;
			remainingBatchSize = inputLayer.remainingBatchSize;
		} else if (model.layerList.get(0) instanceof ConvolutionalLayer) {
			batchSize = 1;
		}

		BackPropagator.weightList = weights.weightList;
		layerCounter = weightList.size();
	}

	protected double[][] computeDerivative(Layer input) {
		double[][] derivative;
		derivative = activator.computeActivatedDerivative(input);
		return derivative;
	}

	protected double[][] removeBiasColumn(double[][] layerValue) {
		double[][] result = new double[layerValue.length][layerValue[0].length - 1];
		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length - 1; j++) {
				result[i][j] = layerValue[i][j];
			}
		}
		return result;
	}

	protected double[][] concatenateColumn(double[][] input) {
		double[][] inputsWithBiases = new double[input.length][input[0].length + 1];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				inputsWithBiases[i][j] = input[i][j];
			}
		}
		for (int i = 0; i < input.length; i++) {
			inputsWithBiases[i][input[0].length] = 1;
		}
		return inputsWithBiases;
	}

}

class DenseBackPropagator extends BackPropagator {

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		gradients = new Gradients();

		// if (nextLayer instanceof HiddenLayer || nextLayer instanceof InputLayer) {
		gradient = nt.matrixMultiplication(previousPartialGradient, nt.matrixTranspose(weightList.get(layerCounter)));
		layer.preActivatedValue = concatenateColumn(layer.preActivatedValue);
		gradient = nt.elementwiseMultiplication(gradient, computeDerivative(layer));
		gradient = removeBiasColumn(gradient);
		previousPartialGradient = gradient;
		if ((nextLayer instanceof HiddenLayer)) {
			gradient = nt.matrixMultiplication(nt.matrixTranspose(nextLayer.layerValue), gradient);
		} else {
			gradient = nt.matrixMultiplication(nt.matrixTranspose(((InputLayer) nextLayer).currentBatch), gradient);
		}

		if (layerCounter == 1) {
			for (int j = 0; j < gradient.length; j++) {
				for (int k = 0; k < gradient[0].length; k++) {
					gradient[j][k] = ((double) remainingBatchSize / (double) batchSize) * gradient[j][k];
				}
			}
		}

		for (int j = 0; j < gradient.length; j++) {
			for (int k = 0; k < gradient[0].length; k++) {
				gradient[j][k] = gradient[j][k] + regularize * weightList.get(layerCounter - 1)[j][k];
			}
		}
		// }

		gradients.twoDGradient = gradient;

		return gradients;
	}

	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		double[][] reformattedArray;
		double[][] preFlattendArray = nt.copyArray(layer.preActivatedValue);

		gradients = new Gradients();

		// if (nextLayer instanceof HiddenLayer || nextLayer instanceof InputLayer) {
		gradient = nt.matrixMultiplication(previousPartialGradient, nt.matrixTranspose(weightList.get(layerCounter)));
		layer.preActivatedValue = layer.layerValue;
		gradient = nt.elementwiseMultiplication(gradient, computeDerivative(layer));
		// gradient = removeBiasColumn(gradient);
		previousPartialGradient = gradient;

		reformattedArray = reformatArray(preFlattendArray, previousPartialGradient);

		gradients.runningTotal = reformattedArray;

		// return gradients;
	}

	private double[][] reformatArray(double[][] preFlattendArray, double[][] previousPartialGradient) {
		int counter = 0;
		double[][] reformattedArray = new double[preFlattendArray.length][preFlattendArray[0].length];
		for (int i = 0; i < preFlattendArray.length; i++) {
			for (int j = 0; j < preFlattendArray[0].length; j++) {
				reformattedArray[i][j] = previousPartialGradient[0][counter];
				counter++;
			}
		}
		return reformattedArray;
	}

}

class OutputBackPropagator extends BackPropagator {

	private double[][] computePartialGradientLastLayer(Layer finalLayer) {
		double[][] partialGradient;
		if (finalLayer.activation.equals("SOFTMAX")) {
			partialGradient = nt.elementwiseMultiplication((computeDerivativeofError(getTargetBatch(), finalLayer)),
					computeDerivative(finalLayer));
		} else {
			partialGradient = nt.elementwiseMultiplication((computeDerivativeofError(getTargetBatch(), finalLayer)),
					nt.scalarMultiply(-1.0, computeDerivative(finalLayer)));
		}
		return partialGradient;
	}

	private double[][] computeDerivativeofError(double[][] targetBatch, Layer finalLayer) {
		double[][] derivativeOfError = new double[targetBatch.length][targetBatch[0].length];
		double[][] finalLayerValue = nt.copyArray(finalLayer.layerValue);

		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < targetBatch.length; i++) {
				for (int j = 0; j < targetBatch[0].length; j++) {

					if (finalLayerValue[i][j] < 0.000000001) {
						finalLayerValue[i][j] += .000000001;
					}
					if (finalLayerValue[i][j] > .999999999) {
						finalLayerValue[i][j] -= .000000001;
					}

					derivativeOfError[i][j] = -1 * (targetBatch[i][j] * (1 / finalLayerValue[i][j]))
							+ ((1.0 - targetBatch[i][j]) * (1.0 / (1.0 - finalLayerValue[i][j])));
				}
			}

		} else {
			for (int i = 0; i < targetBatch.length; i++) {
				for (int j = 0; j < targetBatch[0].length; j++) {
					derivativeOfError[i][j] = targetBatch[i][j] - finalLayerValue[i][j];
				}
			}
		}
		return derivativeOfError;
	}

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		gradients = new Gradients();
		previousPartialGradient = computePartialGradientLastLayer(layer);

		if (layerList.size() > 2) { // must add layer size handling, must have input and output layer (at least two
									// layers)
			gradient = nt.matrixMultiplication(nt.matrixTranspose(nextLayer.layerValue), previousPartialGradient);
		} else {
			gradient = nt.matrixMultiplication(nt.matrixTranspose(nextLayer.currentBatch), previousPartialGradient);
		}

		for (int i = 0; i < gradient.length; i++) {
			for (int j = 0; j < gradient[0].length; j++) {
				gradient[i][j] = gradient[i][j] + regularize * weightList.get(weightList.size() - 1)[i][j];
			}
		}

		gradients.twoDGradient = gradient;

		return gradients;
	}

	public boolean hasReachedEndofBatchTarget() {
		if (nt.numofBatches - 1 == batchCounterTarget) {
			return true;
		} else {
			return false;
		}
	}

	int batchTrackerTarget = 0;
	int batchCounterTarget = 0;
	double[][] currentTargetBatch;

	private double[][] getTargetBatch() {
		double[][] batch;

		if (remainingBatchSize == 0) {
			remainingBatchSize = batchSize;
		}
		if (!hasReachedEndofBatchTarget()) {
			batch = new double[batchSize][targets.targetSize];
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					batch[i][j] = targets.targets[batchTrackerTarget][j];
				}
				batchTrackerTarget++;
			}
			batchCounterTarget++;
			currentTargetBatch = batch;
		} else {
			batch = new double[remainingBatchSize][targets.targetSize];

			for (int i = 0; i < remainingBatchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					batch[i][j] = targets.targets[batchTrackerTarget][j];
				}
				batchTrackerTarget++;
			}
			batchTrackerTarget = 0;
			batchCounterTarget = 0;
			currentTargetBatch = batch;
		}
		
		return batch;
	}

}

class HiddenConvolutionalBackPropagator extends BackPropagator {
	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		gradients = new Gradients();
		// return gradients;
	}

	/*
	 * public List<double[][][]> propagate(Layer layer, Layer nextLayer) {
	 * ConvolutionalLayer
	 * 
	 * double[][] input = layer.layerValue; int tracker = 0; int strideLength =
	 * conv.strideLength; int numofXCycles = (input[0].length -
	 * filterSize)/strideLength + 1; int numofYCycles = (input.length -
	 * filterSize)/strideLength + 1; double convOutput = 0;
	 * 
	 * double[][] convOutputArraySum = new double[numofYCycles][numofXCycles];
	 * double[][] convOutputArrayAugmented = new double[numofYCycles *
	 * filters.size()][numofXCycles];
	 * 
	 * for (int h = 0; h < filters.size(); h++) { for (int cornerPosY = 0;
	 * cornerPosY < numofYCycles; cornerPosY += strideLength) { for (int cornerPosX
	 * = 0; cornerPosX < numofXCycles; cornerPosX += strideLength) { for (int j = 0;
	 * j < filterSize; j++) { for (int k = 0; k < filterSize; k++) { convOutput +=
	 * input[j + cornerPosY][k + cornerPosX] * filters.get(h)[j][k]; } }
	 * convOutputArraySum[cornerPosY / strideLength][cornerPosX / strideLength] =
	 * convOutput + biasTerm; convOutput = 0; } }
	 * 
	 * for (int i = 0; i < numofYCycles; i++) { for (int j = 0; j < numofXCycles;
	 * j++) { convOutputArrayAugmented[i + tracker * numofYCycles][j] =
	 * convOutputArraySum[i][j]; } } tracker++; }
	 * 
	 * nextLayer.layerValue = convOutputArrayAugmented;
	 * 
	 * System.out.print("Layer 1 "); for (int i = 0; i < 5; i++) { for (int j = 0; j
	 * < 5; j++) { System.out.print(convOutputArrayAugmented[i][j] + " "); } }
	 * System.out.println();
	 * 
	 * return convOutputArrayAugmented; }
	 * 
	 * 
	 */

	private double[][] flipBoard(double[][] board) {
		double[][] newBoard = new double[board.length][board.length];
		for (int i = 0; i < board.length; i++) {
			for (int j = 0; j < board.length; j++) {
				newBoard[i][j] = board[(board.length - 1) - i][(board.length - 1) - j];
			}
		}
		return newBoard;
	}
}

class PoolingBackPropagator extends BackPropagator {
	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		double[][] dPool = inflateArray((PoolingLayer) layer, gradients.runningTotal);
		dPool = nt.elementwiseMultiplication(dPool, ((PoolingLayer) layer).expandedLayer);
		gradients.runningTotal = dPool;
		// return gradients;
	}

	private double[][] inflateArray(PoolingLayer layer, double[][] array) {
		int squareSize = layer.poolSize;
		int totalSizeY = squareSize * array.length;
		int totalSizeX = squareSize * array[0].length;
		double[][] newArray = new double[totalSizeY][totalSizeX];

		for (int i = 0; i < totalSizeY; i += squareSize) {
			for (int j = 0; j < totalSizeX; j += squareSize) {
				for (int k = 0; k < squareSize; k++) {
					for (int l = 0; l < squareSize; l++) {
						newArray[i + k][j + l] = array[i / squareSize][j / squareSize];
					}
				}
			}
		}
		return newArray;
	}
}

class ReluBackPropagator extends BackPropagator {
	List<double[][]> filterList = new ArrayList<double[][]>();

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		double[][] filter = nt.elementwiseMultiplication(gradients.runningTotal, computeDerivative(layer));
		double[][][] image;
		
		ConvolutionalLayer conv = (ConvolutionalLayer) nextLayer;
		int numofFilters = conv.numofFilters;

		filterList = (splitFilters(filter, numofFilters));

		filterList = rotate90(rotate90(filterList));

		conv = (ConvolutionalLayer) nextLayer;

		image = conv.currentImage;

		int strideLengthY = 1;
		int strideLengthX = 1; 

		List<double[][][]> outputList = new ArrayList<double[][][]>();

		for (double[][] filter1 : filterList) { // loop through filters
			double[][][] output = new double[image.length][(image[0].length - filter1.length)/strideLengthY + 1][(image[0][0].length - filter1[0].length)/strideLengthX + 1]; // TODO: add actual values
			for (int i = 0; i < image.length; i++) { // loop through rgb values
				for (int j = 0; j < (image[0].length - filter1.length)/strideLengthY + 1; j += strideLengthY) {
					for (int k = 0; k < (image[0][0].length - filter1[0].length)/strideLengthX + 1; k += strideLengthX) {
						double total = 0;
						for (int l = 0; l < filter1.length; l++) { // loop through filter
							for (int m = 0; m < filter1[0].length; m++) { // loop through filter
								total += filter1[l][m] * image[i][j + l][k + m];
							}
						}
						output[i][j / strideLengthY][k / strideLengthX] = total;
					}
				}
			}
			outputList.add(output);
		}

		gradients = new Gradients();

		gradients.threeDGradientList = outputList;
		
		//System.out.println(outputList.get(0)[0].length);

		return gradients;
	}

	private List<double[][]> splitFilters(double[][] input, int numofFilters) {
		int counter = 0;
		int filterSizeY = input.length/numofFilters;
		int filterSizeX = input[0].length;
		
		double[][] array;
		List<double[][]> list = new ArrayList<double[][]>();

		for (int i = 0; i < numofFilters; i++) {
			array = new double[filterSizeY][filterSizeX];
			for (int j = 0; j < array.length; j++) {
				for (int k = 0; k < array[0].length; k++) {
					array[j][k] = input[j + filterSizeY * counter][k];
				}
			}
			list.add(array);
			counter++;
		}
		return list;
	}

	private List<double[][]> rotate90(List<double[][]> mat) {
		int M = mat.get(0).length;
		int N = mat.get(0)[0].length;
		double[][] ret;
		List<double[][]> list = new ArrayList<double[][]>();

		for (int i = 0; i < mat.size(); i++) {
			ret = new double[N][M];
			for (int r = 0; r < M; r++) {
				for (int c = 0; c < N; c++) {
					ret[c][M - 1 - r] = mat.get(i)[r][c];
				}
			}
			list.add(ret);
		}
		return list;
	}

	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		nextLayer.activation = "RELU"; //shouldn't have to worry
		gradient = computeDerivative(nextLayer);

		if (gradients.runningTotal == null) {
			gradients.runningTotal = gradient;
		} else {
			gradients.runningTotal = nt.elementwiseMultiplication(gradients.runningTotal, gradient);
		}

	}

}
