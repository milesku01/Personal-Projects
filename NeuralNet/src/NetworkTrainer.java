import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class NetworkTrainer {
	int batchSize = 0;
	static int numofLayers; 
	int convBatchSize = 0;
	int numofEpochs = 0;
	static int remainingBatchSize;
	List<Layer> layers;

	InputLayer inputLayer;
	ConvolutionalLayer convLayer;
	Targets targets;
	static List<double[][]> weightList;
	List<Filters> filterList;
	Activator activator = new Activator();
	Optimizer optimizer = new Optimizer();
	ForwardPropagator fp = new ForwardPropagator();
	BackPropagator bp = new BackPropagator();
	Layer layer = new Layer();
	NetworkEvaluator eval;
	FileReader fr; 
	Weights weights;
	double[][] batchPart;
	static double accuracy = 0;
	static int numofBatches;
	String optimizerString;
	static String[] activatorStrings;

	public void train(NetworkModel model, int numofEpochs, String optimizerString) {
		this.numofEpochs = numofEpochs;
		layers = model.layerList;
		numofLayers = layers.size();

		weights = new Weights();

		weights.generateInitialWeights(model);

		if (layers.get(0) instanceof InputLayer) {
			inputLayer = (InputLayer) layers.get(0);
			batchSize = inputLayer.batchSize;
			numofBatches = calculateNumofBatches();
			if (layers.get(0).layerValue.length % batchSize != 0) {
				remainingBatchSize = layers.get(0).layerValue.length % batchSize;
			} else {
				remainingBatchSize = batchSize;
			}

		} else if (layers.get(0) instanceof ConvolutionalLayer) {
			convLayer = (ConvolutionalLayer) layers.get(0);
			filterList = weights.filterList;
			numofBatches = layers.get(0).globalNumofSets;

			if (numofBatches > 90) {
				numofBatches *= .7;
			}
			convBatchSize = convLayer.batchSize;
			batchSize = 1;
			remainingBatchSize = 1;
		}

		activatorStrings = new String[layers.size() - 1];
		weightList = weights.weightList;
		targets = model.targets;
		this.optimizerString = optimizerString;

		getActivatorStrings();

		int iterations = 1;
		if (layers.get(0) instanceof ConvolutionalLayer) {
			iterations = (numofBatches / convBatchSize) * numofEpochs;
		} else {
			iterations = numofBatches * numofEpochs;
		}

		// iterations = 1;

		fp.constructForwardPropagationObjects(layers, weights);
		bp.constructBackwardPropagationObjects(model, weights);

		long startTime = System.nanoTime();

		for (int i = 1; i <= iterations + 1; i++) {

			if (layers.get(0) instanceof InputLayer) {

				forwardPropagation();

				if (i != iterations + 1)
					backPropagation();
			} else if (layers.get(0) instanceof ConvolutionalLayer) {

				for (int j = 0; j < convBatchSize; j++) {
					forwardPropagation();

					if (i != iterations + 1)
						convolutionalBackPropagation();
				}
				if (convBatchSize != 1) {
					averageGradients();
				}
				weightChanges = optimizer.optimize(gradientCopy, optimizerString);
				updateParameters();
				cleanGradCopy();
			}

			formatOutput(i);
		}

		if (layers.get(0).globalNumofSets > 90) {
			determineAccuracy();
		}

		long endTime = System.nanoTime();
		System.out.println(" \nTraining time: " + getTrainingTime(startTime, endTime) + " sec");
	}

	NetworkTrainer net;

	public void trainUntil(NetworkModel model, double accuracy, int numofEpochs, String optimizerString) {

		if (model.layerList.get(model.layerList.size() - 1).activation.equals("SOFTMAX")) {
			while (this.accuracy < accuracy) {
				net = new NetworkTrainer();
				net.train(model, numofEpochs, optimizerString);
			}
		} else {
			do {
				net = new NetworkTrainer();
				net.train(model, numofEpochs, optimizerString);
			} while (this.accuracy < accuracy);
		}
	}
	
	public void trainUntilMatchFound(NetworkModel model, int numofEpochs, String matchingTable, String optimizerString, String lookup) {
		eval = new NetworkEvaluator(); 

		do {
			net = new NetworkTrainer();
			net.train(model, numofEpochs, optimizerString);
		} while (!eval.pairsMatch(inputLayer.normalizer, numofLayers, weightList, matchingTable, lookup));
	}
	
	
	
	

	private void getActivatorStrings() {
		for (int i = 0; i < activatorStrings.length; i++) {
			activatorStrings[i] = layers.get(i + 1).activation;
		}
	}

	public String getTrainingTime(long startTime, long endTime) {
		Double n = (double) (endTime - startTime) / (double) 1000000000;
		DecimalFormat df = new DecimalFormat("#.####");
		df.setRoundingMode(RoundingMode.CEILING);
		Double d = n.doubleValue();
		return df.format(d);
	}

	public int calculateNumofBatches() {
		double rawBatchNum = Math.ceil((double) layers.get(0).layerValue.length / (double) batchSize);
		return (int) rawBatchNum;
	}

	public void printArray(double[][] d) {
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[0].length; j++) {
				System.out.print(Math.round(d[i][j] * 10000) / 10000.0 + " ");
			}
		}
		System.out.println();
	}

	public void print3DArray(double[][][] d) {
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[0].length; j++) {
				for (int k = 0; k < d[0][0].length; k++) {
					System.out.print(Math.round(d[i][j][k] * 10000) / 10000.0 + " ");
				}
			}
		}
		System.out.println();
	}

	public void forwardPropagation() {
		for (int i = 0; i < layers.size() - 1; i++) {
			if (layers.get(i) instanceof InputLayer || layers.get(i) instanceof HiddenLayer
					|| layers.get(i) instanceof DropoutLayer) {
				layers.get(i + 1).layerValue = fp.propagate(layers.get(i), layers.get(i + 1)); // nextLayer,
																								// previousLayer
			} else {
				layers.get(i + 1).convValue = fp.propagateConv(layers.get(i), layers.get(i + 1)); // nextLayer,
																									// previousLayer
			}
		}

		//System.out.print("OutputArray ");
		//printArray(layers.get(layers.size() - 1).layerValue);
	}

	int targetPositionCounter = 0;
	double[][] targetPart;

	private void determineTargetForError() {

		if (targetPositionCounter != (batchSize * (numofBatches - 1))) {
			batchPart = new double[batchSize][targets.targetSize];
			targetPart = new double[batchSize][targets.targetSize];

			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					batchPart[i][j] = layers.get(layers.size() - 1).layerValue[i][j];
					targetPart[i][j] = targets.targets[i + targetPositionCounter][j];
				}
			}
			targetPositionCounter += batchSize;
		} else {
			batchPart = new double[remainingBatchSize][targets.targetSize];
			targetPart = new double[batchSize][targets.targetSize];

			for (int i = 0; i < remainingBatchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					batchPart[i][j] = layers.get(layers.size() - 1).layerValue[i][j];
					targetPart[i][j] = targets.targets[i + targetPositionCounter][j];
				}
			}
			targetPositionCounter = 0;
		}
	}

	double[][] finalTestLayer;
	double[][] layerValue;

	private void determineAccuracy() {
		finalTestLayer = forwardPropTest();
		if (activatorStrings[activatorStrings.length - 1].equals("SOFTMAX")) {
			System.out.println("Accuracy " + computeAccuracy());
		} else {
			System.out.println("Loss over test set " + computeAccuracy());
		}
	}

	public double[][] forwardPropTest() {
		double[][] returnValue = null;

		if (layers.get(0) instanceof InputLayer) {
			for (int i = 0; i < layers.size() - 1; i++) {
				layers.get(i + 1).layerValue = fp.propagateTest(layers.get(i), layers.get(i + 1));
			}
			returnValue = layers.get(layers.size() - 1).layerValue;

		} else if (layers.get(0) instanceof ConvolutionalLayer) {
			returnValue = new double[((ConvolutionalLayer) layers.get(0)).testingImages.size()][targets.targetSize];

			for (int i = 0; i < ((ConvolutionalLayer) layers.get(0)).testingImages.size(); i++) {
				for (int j = 0; j < layers.size() - 1; j++) {

					if (layers.get(j) instanceof InputLayer || layers.get(j) instanceof HiddenLayer
							|| layers.get(j) instanceof DropoutLayer) {
						layers.get(j + 1).layerValue = fp.propagateTest(layers.get(j), layers.get(j + 1));
					} else {
						layers.get(j + 1).convValue = fp.propagateConvTest(layers.get(j), layers.get(j + 1));
					}
				}
				for (int k = 0; k < targets.targetSize; k++) {
					returnValue[i][k] = layers.get(layers.size() - 1).layerValue[0][k];
				}
			}
		}

		return returnValue;
	}

	public void formatOutput(int i) {
		// if (i % numofBatches == 0) {
		System.out.println();

		// System.out.println("Current batch " +
		// java.util.Arrays.deepToString(layers.get(0).currentBatch));

		// for (int j = 0; j < layers.size() - 1; j++) {
		// System.out.println(layers.get(i).layerValue.length + " " +
		// layers.get(i).layerValue[0].length);
		// }
		// System.out.println("Last Layer " +
		// java.util.Arrays.deepToString(layers.get(layers.size()-1).layerValue));

		// System.out.println("Targets: " +
		// java.util.Arrays.deepToString(targets.targets));
		// System.out.println();
		// for (int j = 0; j < weightList.size(); j++) {
		// System.out.println("weight " + j +
		// java.util.Arrays.deepToString(weightList.get(j)));
		// }

		System.out.println("Loss: " + reportLoss(layers.get(layers.size() - 1))); //
		// }
	}

	private double computeAccuracy() {
		int correct = 0;
		double loss = 0;
		double[][] result = copyArray(finalTestLayer);
		double[][] target = copyArray(targets.testTargets);
		String lastAct = activatorStrings[activatorStrings.length - 1];

		if (lastAct.equals("SOFTMAX")) {
			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {
					if (target[i][j] != 0.0) {
						if (Math.abs(target[i][j] - result[i][j]) < .5) {
							correct++;
						}
					}
				}
			}
		} else {

			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {
					loss += Math.pow((target[i][j] - result[i][j]), 2);
				}
			}
			loss *= (.5 / ((double) result.length * (double) result[0].length));

			if (result[0].length == 2) {
				for (int i = 0; i < result.length; i++) {
					if (((result[i][0] > result[i][1]) && (target[i][0] > target[i][1]))
							|| ((result[i][0] < result[i][1]) && (target[i][0] < target[i][1]))) {
						correct++;
					}
				}
				System.out.println("Accuracy " + (double) (correct / (double) result.length));
			}

		}

		if (lastAct.equals("SOFTMAX")) {
			accuracy = (double) (correct / (double) result.length);
			return accuracy; // must change
		} 
		
		else {
			accuracy = (double) (correct / (double) result.length);
			return loss;
		}

	}

	public double reportLoss(Layer finalLayer) {
		double loss = 0;
		double[][] result = copyArray(batchPart);
		double[][] target = copyArray(targetPart);

		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {

					if (result[i][j] == 0.0) {
						result[i][j] += .000000001;
					} else if (result[i][j] == 1.0) {
						result[i][j] -= .000000001;
					}

					// loss += ((target[i][j] * Math.log(result[i][j]))
					// + ((1.0 - target[i][j]) * Math.log(1.0 - result[i][j])));

					loss += (target[i][j] * Math.log(result[i][j]));
				}
			}
			loss *= (-1.0 / (double) result.length);
		} else {
			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {
					loss += Math.pow((target[i][j] - result[i][j]), 2);
				}
			}
			loss *= (.5 / ((double) result.length * (double) result[0].length));
		}
		System.out.println("Reg " + regularizationTerm());
		loss += regularizationTerm();
		return loss;
	}

	private double regularizationTerm() {
		double currentLength = 1;
		double weightSquaredSum = 0.0;

		if (layers.get(0) instanceof InputLayer) {
			currentLength = inputLayer.currentBatch.length;
		} else if (layers.get(0) instanceof ConvolutionalLayer) {
			currentLength = 1; // I think this is true because of gradient averaging
		}

		for (int i = 0; i < weightList.size(); i++) {
			for (int j = 0; j < weightList.get(i).length; j++) {
				for (int k = 0; k < weightList.get(i)[0].length; k++) {
					weightSquaredSum += Math.pow(weightList.get(i)[j][k], 2);
				}
			}
		}
		weightSquaredSum *= bp.regularize / (2.0 * (double) currentLength);
		return weightSquaredSum;
	}

	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer);
		return activatedValue;
	}

	List<Object> weightChanges;

	public void backPropagation() {
		determineTargetForError();
		computeGradients();
		weightChanges = optimizer.optimize(gradients, optimizerString);
		updateParameters();
		cleanGradients();
	}

	boolean TorF = true;

	public void convolutionalBackPropagation() {
		determineTargetForError();
		computeGradients();

		if (TorF) {
			initializeGradientCopy();
			TorF = false;
		}
		sumGradients();
		cleanGradients();
	}

	private void computeGradients() {
		Gradients gradient = null;
		for (int i = layers.size() - 1; i > 0; i--) {
			gradient = bp.computeGradients(layers.get(i), layers.get(i - 1));

			if ((layers.get(i - 1) instanceof ReluLayer == false)
					&& (layers.get(i - 1) instanceof PoolingLayer == false)) {
				gradients.add(gradient);
			}
		}

	}

	private void sumGradients() {
		for (int i = 0; i < gradients.size(); i++) {
			if (gradients.get(i).twoDGradient != null) {
				for (int j = 0; j < gradients.get(i).twoDGradient.length; j++) {
					for (int k = 0; k < gradients.get(i).twoDGradient[0].length; k++) {
						gradientCopy.get(i).twoDGradient[j][k] += gradients.get(i).twoDGradient[j][k];
					}
				}
			} else if (gradients.get(i).twoDGradient == null) { // must change when adding other 2d filters
				for (int h = 0; h < gradients.get(i).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradients.get(i).threeDGradientList.get(h).length; l++) {
						for (int j = 0; j < gradients.get(i).threeDGradientList.get(h)[0].length; j++) {
							for (int k = 0; k < gradients.get(i).threeDGradientList.get(h)[0][0].length; k++) {
								gradientCopy.get(i).threeDGradientList
										.get(h)[l][j][k] += gradients.get(i).threeDGradientList.get(h)[l][j][k];
							}
						}
					}
				}
			}
		}
	}

	private void averageGradients() {
		for (int i = 0; i < gradients.size(); i++) {
			if (gradients.get(i).twoDGradient != null) {
				for (int j = 0; j < gradients.get(i).twoDGradient.length; j++) {
					for (int k = 0; k < gradients.get(i).twoDGradient[0].length; k++) {
						gradientCopy.get(i).twoDGradient[j][k] /= convBatchSize;
					}
				}
			} else if (gradients.get(i).twoDGradient == null) { // must change when adding other 2d filters
				for (int h = 0; h < gradients.get(i).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradients.get(i).threeDGradientList.get(h).length; l++) {
						for (int j = 0; j < gradients.get(i).threeDGradientList.get(h)[0].length; j++) {
							for (int k = 0; k < gradients.get(i).threeDGradientList.get(h)[0][0].length; k++) {
								gradientCopy.get(i).threeDGradientList.get(h)[l][j][k] /= convBatchSize;
							}
						}
					}
				}
			}
		}
	}

	List<Gradients> gradients = new ArrayList<Gradients>();
	List<Gradients> gradientCopy = new ArrayList<Gradients>();

	private void initializeGradientCopy() {
		double[][] twoArray;
		double[][][] threeArray;
		List<double[][][]> threeArrayList;
		Gradients newGradient;

		for (int i = 0; i < gradients.size(); i++) {
			newGradient = new Gradients();
			if (gradients.get(i).twoDGradient != null) {
				twoArray = new double[gradients.get(i).twoDGradient.length][gradients.get(i).twoDGradient[0].length];
				newGradient.twoDGradient = twoArray;
				gradientCopy.add(newGradient);

			} else if (gradients.get(i).twoDGradient == null) {
				threeArrayList = new ArrayList<double[][][]>();
				for (int j = 0; j < gradients.get(i).threeDGradientList.size(); j++) {
					threeArray = new double[gradients.get(i).threeDGradientList
							.get(j).length][gradients.get(i).threeDGradientList
									.get(j)[0].length][gradients.get(i).threeDGradientList.get(j)[0][0].length];
					threeArrayList.add(threeArray);
				}
				newGradient.threeDGradientList = threeArrayList;
				gradientCopy.add(newGradient);
			}
		}
	}

	private void cleanGradients() {
		gradients.clear();
	}

	private void cleanGradCopy() {
		gradientCopy.clear();
		TorF = true;
	}

	int weightListCounter = 0;
	int filterListCounter = 0; // for later

	private void updateParameters() {

		for (int i = 0; i < weightChanges.size(); i++) {
			if (weightChanges.get(i) instanceof double[][]) {
				for (int j = 0; j < ((double[][]) weightChanges.get(i)).length; j++) {
					for (int k = 0; k < ((double[][]) weightChanges.get(i))[0].length; k++) {
						weightList.get(weightListCounter)[j][k] -= ((double[][]) weightChanges.get(i))[j][k];
					}
				}
				weightListCounter++;
			} else if (weightChanges.get(i) instanceof List) { // must change when adding other 2d filters
				for (int h = 0; h < ((List<double[][][]>) weightChanges.get(i)).size(); h++) {
					for (int l = 0; l < ((List<double[][][]>) weightChanges.get(i)).get(h).length; l++) {
						for (int j = 0; j < ((List<double[][][]>) weightChanges.get(i)).get(h)[0].length; j++) {
							for (int k = 0; k < ((List<double[][][]>) weightChanges.get(i)).get(h)[0][0].length; k++) {
								filterList.get(filterListCounter).threeDFilterArray
										.get(h)[l][j][k] -= ((List<double[][][]>) weightChanges.get(i)).get(h)[l][j][k];
							}
						}
					}
				}
				filterListCounter++;
			}
		}
		weightListCounter = 0;
		filterListCounter = 0;
	}

	double[][] matrixMultiplication(double[][] A, double[][] B) {
		int aRows = A.length;
		int aColumns = A[0].length;
		int bRows = B.length;
		int bColumns = B[0].length;
		double[][] C = new double[aRows][bColumns];

		if (aColumns != bRows) {
			throw new IllegalArgumentException("A:col: " + aColumns + " did not match B:rows " + bRows + ".");
		}

		for (int i = 0; i < aRows; i++) { // aRow
			for (int j = 0; j < bColumns; j++) { // bColumn
				for (int k = 0; k < aColumns; k++) { // aColumn
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		return C;
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

	public double[][] appendBiasColumn(double[][] layer) {
		double[][] inputsWithBiases = new double[layer.length][layer[0].length + 1];

		for (int i = 0; i < layer.length; i++) {
			for (int j = 0; j < layer[0].length; j++) {
				inputsWithBiases[i][j] = layer[i][j];
			}
		}
		for (int i = 0; i < layer.length; i++) {
			inputsWithBiases[i][layer[0].length] = 1;
		}
		return inputsWithBiases;
	}

	public double[][] elementwiseMultiplication(double[][] m, double[][] n) {
		double[][] result = new double[m.length][m[0].length];
		if (m.length != n.length || m[0].length != n[0].length) {
			throw new IllegalArgumentException(
					"Dimensions did not match" + m.length + " " + m[0].length + " " + n.length + " " + n[0].length);
		}
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[0].length; j++) {
				result[i][j] = m[i][j] * n[i][j];
			}
		}
		return result;
	}

	public double[][] matrixTranspose(double[][] m) {
		double[][] temp = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	public double[][] scalarMultiply(double num, double[][] m) {
		for (int i = 0; i < m.length; i++) {
			for (int j = 0; j < m[0].length; j++) {
				m[i][j] = num * m[i][j];
			}
		}
		return m;
	}

}
