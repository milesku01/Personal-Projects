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
		
		if ((layer instanceof PoolingLayer) || ((layer instanceof HiddenLayer || layer instanceof DropoutLayer || layer instanceof HiddenConvolutionalLayer)
						&& (nextLayer instanceof PoolingLayer || nextLayer instanceof ReluLayer))) {
			propagationObjects.get(objectTracker).computeHiddenGradients(layer, nextLayer);
			
		} else {
			gradients = propagationObjects.get(objectTracker).computeGradients(layer, nextLayer);
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
			} else if(layerList.get(i) instanceof DropoutLayer) {
				backPropObj = new DenseBackPropagator();
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

	public double[][][] threeDElementMultiplication(double[][][] input, double[][][] input2) {

		if (input.length != input2.length || input[0].length != input2[0].length
				|| input[0][0].length != input2[0][0].length) {
			throw new IllegalArgumentException("Dimensions did not match" + input.length + " " + input2.length + " "
					+ input[0].length + " " + input2[0].length + " " + input[0][0].length + " " + input2[0][0].length);
		}

		double[][][] output = new double[input.length][input[0].length][input[0][0].length];

		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				for (int k = 0; k < input[0][0].length; k++) {
					output[i][j][k] = input[i][j][k] * input2[i][j][k];
				}
			}
		}
		return output;

	}

}

class DenseBackPropagator extends BackPropagator {

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		gradients = new Gradients();

		gradient = nt.matrixMultiplication(previousPartialGradient, nt.matrixTranspose(weightList.get(layerCounter)));
		layer.preActivatedValue = concatenateColumn(layer.preActivatedValue);
		gradient = nt.elementwiseMultiplication(gradient, computeDerivative(layer));
		gradient = removeBiasColumn(gradient);
		previousPartialGradient = gradient;
		
		if ((nextLayer instanceof HiddenLayer) || (nextLayer instanceof DropoutLayer)) {
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

		gradients.twoDGradient = gradient;

		return gradients;
	}

	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		double[][][] reformattedArray;
		double[][][] preFlattendArray = copyThreeDArray(layer.preActivatedConvValue);

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

	private double[][][] reformatArray(double[][][] preFlattendArray, double[][] previousPartialGradient) {
		int counter = 0;
		double[][][] reformattedArray = new double[preFlattendArray.length][preFlattendArray[0].length][preFlattendArray[0][0].length];
		for (int i = 0; i < preFlattendArray.length; i++) {
			for (int j = 0; j < preFlattendArray[0].length; j++) {
				for (int k = 0; k < preFlattendArray[0][0].length; k++) {
					reformattedArray[i][j][k] = previousPartialGradient[0][counter];
					counter++;
				}
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

	//	System.out.print("TARGETBATCH ");
	//	nt.printArray(batch);

		return batch;
	}

}

class HiddenConvolutionalBackPropagator extends BackPropagator {

	List<double[][][]> filterList;

	public void computeHiddenGradients(Layer layer, Layer nextLayer) { 

		HiddenConvolutionalLayer hidden = (HiddenConvolutionalLayer) layer;

		double[][][] dOut = hidden.fullyConvolvedDerivative;

		filterList = copyList(hidden.filterList); // may need to copy

		filterList = rotate90(rotate90(filterList));

		dOut = pad(dOut, filterList.get(0)[0].length);

		int strideLengthY = 1;
		int strideLengthX = 1;
		double total = 0;
		
		double[][][] output = new double[filterList.get(0).length][dOut[0].length - filterList.get(0)[0].length + 1][dOut[0][0].length - filterList.get(0)[0][0].length + 1];

		for (int i = 0; i < filterList.get(0).length; i++) {
			for (int j = 0; j < filterList.size(); j++) {
				for (int k = 0; k < (dOut[0].length - filterList.get(j)[0].length) / strideLengthY + 1; k += strideLengthY) {
					for (int l = 0; l < (dOut[0][0].length - filterList.get(j)[0][0].length) / strideLengthX + 1; l += strideLengthX) {
						total = 0;
						for (int m = 0; l < filterList.get(j)[0].length; l++) { // loop through filter
							for (int n = 0; m < filterList.get(j)[0][0].length; m++) { // loop through filter
								total += filterList.get(j)[i][m][n] * dOut[j][k + m][l + n];
							}
						}
						output[i][k / strideLengthY][l / strideLengthX] += total;
					}
				}
			}
		}

		gradients.runningTotal = output; // okay because gradient is reset just before, thats why no multiplication

	}

	private double[][][] pad(double[][][] input, int padSize) {
		int padding = padSize - 1;

		double[][][] output = new double[input.length][input[0].length + 2 * padding][input[0][0].length + 2 * padding];

		for (int k = 0; k < input.length; k++) {
			for (int i = 0; i < input[0].length; i++) {
				for (int j = 0; j < input[0][0].length; j++) {
					output[k][i + padding][j + padding] = input[k][i][j];
				}
			}
		}

		return output;
	}

	private List<double[][][]> rotate90(List<double[][][]> mat) {
		int M = mat.get(0)[0].length;
		int N = mat.get(0)[0][0].length;
		double[][][] ret;
		List<double[][][]> list = new ArrayList<double[][][]>(mat.size());

		for (int i = 0; i < mat.size(); i++) {
			ret = new double[mat.get(0).length][N][M];
			for (int j = 0; j < mat.get(0).length; j++) {
				for (int r = 0; r < M; r++) {
					for (int c = 0; c < N; c++) {
						ret[j][c][M - 1 - r] = mat.get(i)[j][r][c];
					}
				}
			}
			list.add(ret);
		}
		return list;
	}

	private List<double[][][]> copyList(List<double[][][]> input) {
		List<double[][][]> copy = new ArrayList<double[][][]>(input.size());
		for (int i = 0; i < input.size(); i++) {
			copy.add(copyThreeDArray(input.get(i)));
		}
		return copy;
	}

}

class PoolingBackPropagator extends BackPropagator {
	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		double[][][] dPool = inflateArray((PoolingLayer) layer, gradients.runningTotal);
		dPool = threeDElementMultiplication(dPool, ((PoolingLayer) layer).expandedLayer);
		gradients.runningTotal = dPool;
	}

	private double[][][] inflateArray(PoolingLayer layer, double[][][] array) {
		int squareSize = layer.poolSize;
		int totalSizeY = squareSize * array[0].length;
		int totalSizeX = squareSize * array[0][0].length;

		double[][][] newArray = new double[layer.convValue.length][totalSizeY][totalSizeX];

		for (int m = 0; m < layer.convValue.length; m++) {
			for (int i = 0; i < totalSizeY; i += squareSize) {
				for (int j = 0; j < totalSizeX; j += squareSize) {

					for (int k = 0; k < squareSize; k++) {
						for (int l = 0; l < squareSize; l++) {

							newArray[m][i + k][j + l] = array[m][i / squareSize][j / squareSize];
						}
					}
				}
			}
		}
		return newArray;
	}
}

class ReluBackPropagator extends BackPropagator {
	
	List<double[][]> filterList;

	public Gradients computeGradients(Layer layer, Layer nextLayer) {
		double[][][] filter = threeDElementMultiplication(gradients.runningTotal, computeReluDerivative(layer));
		
		double[][][] threeDImage;
		double[][][] image;

		if (nextLayer instanceof ConvolutionalLayer) {

			ConvolutionalLayer conv = (ConvolutionalLayer) nextLayer;

			image = conv.currentImage;

			filterList = (splitFilters(filter));
			
			//filterList = rotate90(rotate90(filterList)); 

			int strideLengthY = 1;
			int strideLengthX = 1;

			List<double[][][]> outputList = new ArrayList<double[][][]>(filterList.size());

			for (double[][] filter1 : filterList) { // loop through filters

				double[][][] output = new double[image.length][(image[0].length - filter1.length) / strideLengthY
						+ 1][(image[0][0].length - filter1[0].length) / strideLengthX + 1]; // TODO: add actual values
				for (int i = 0; i < image.length; i++) { // loop through rgb values
					for (int j = 0; j < (image[0].length - filter1.length) / strideLengthY + 1; j += strideLengthY) {
						for (int k = 0; k < (image[0][0].length - filter1[0].length) / strideLengthX
								+ 1; k += strideLengthX) {
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

		} else if (nextLayer instanceof HiddenConvolutionalLayer) {

			HiddenConvolutionalLayer hiddenConv = (HiddenConvolutionalLayer) nextLayer;

			filterList = splitFilters(filter);

		//	filterList = rotate90(rotate90(filterList)); // doesn't seem to affect anything

			hiddenConv.fullyConvolvedDerivative = copyThreeDArray(filter);

			threeDImage = hiddenConv.convValue;

			int strideLengthY = 1;
			int strideLengthX = 1;

			List<double[][][]> outputList = new ArrayList<double[][][]>(filterList.size());

			for (double[][] filter1 : filterList) { // loop through filters

				double[][][] output = new double[threeDImage.length][(threeDImage[0].length - filter1.length)
						/ strideLengthY + 1][(threeDImage[0][0].length - filter1[0].length) / strideLengthX + 1]; // TODO:
																												
				for (int i = 0; i < threeDImage.length; i++) { // loop through rgb values
					for (int j = 0; j < (threeDImage[0].length - filter1.length) / strideLengthY + 1; j += strideLengthY) {
						for (int k = 0; k < (threeDImage[0][0].length - filter1[0].length) / strideLengthX + 1; k += strideLengthX) {
							double total = 0;
							for (int l = 0; l < filter1.length; l++) { // loop through filter
								for (int m = 0; m < filter1[0].length; m++) { // loop through filter
									total += filter1[l][m] * threeDImage[i][j + l][k + m];
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
		}

		// System.out.println(outputList.get(0)[0].length);

		return gradients;
	}

	public void computeHiddenGradients(Layer layer, Layer nextLayer) {
		nextLayer.activation = "RELU"; // shouldn't have to worry gradient =
		

		if (nextLayer instanceof ConvolutionalLayer || nextLayer instanceof HiddenConvolutionalLayer) {
			gradients.runningTotal = computeReluDerivative(nextLayer);
		} else {
			gradients.runningTotal = threeDElementMultiplication(gradients.runningTotal, computeReluDerivative(nextLayer));
		}

	}

	private double[][][] computeReluDerivative(Layer layer) {
		double[][][] layerValue = copyThreeDArray(layer.preActivatedConvValue);

		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				for (int k = 0; k < layerValue[0][0].length; k++) {

					if (layerValue[i][j][k] < 0) {
						layerValue[i][j][k] = 0;
					} else {
						layerValue[i][j][k] = 1;
					}
				}
			}
		}
		return layerValue;
	}

	private List<double[][]> splitFilters(double[][][] filter) {
		double[][] part = new double[filter[0].length][filter[0][0].length];
		List<double[][]> list = new ArrayList<double[][]>(filter.length);

		for (int i = 0; i < filter.length; i++) {
			for (int j = 0; j < filter[0].length; j++) {
				for (int k = 0; k < filter[0][0].length; k++) {
					part[j][k] = filter[i][j][k];
				}
			}
			list.add(part);
		}
		return list;
	}

	private List<double[][]> rotate90(List<double[][]> mat) {
		int M = mat.get(0).length;
		int N = mat.get(0)[0].length;
		double[][] ret;
		List<double[][]> list = new ArrayList<double[][]>(mat.size());

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
	
}
