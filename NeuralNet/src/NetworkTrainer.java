import java.util.ArrayList;
import java.util.List;

public class NetworkTrainer {
	int batchSize = 0;
	int numofEpochs = 0;
	List<Layer> layers;
	Targets targets;
	List<double[][]> weightList;
	Activator activator = new Activator();
	double[][] currentBatch;

	public void train(NetworkModel model, Weights weights) {
		batchSize = model.batchSize;
		numofEpochs = model.numofEpochs;
		layers = model.layerList;
		weightList = weights.weightList;
		targets = model.targets;

		int iterations = calculateNumofBatches() * numofEpochs;

		for (int i = 0; i < iterations; i++) {
			forwardPropagation();
			backPropagation();
			formatOutput();
		}
	}

	private int calculateNumofBatches() {
		double rawBatchNum = Math.ceil((double) layers.get(0).layerValue.length / (double) batchSize);
		return (int) rawBatchNum;
	}

	public void forwardPropagation() {
		double[][] preActivatedValue;
		propagateInputLayer();
		for (int i = 1; i < layers.size() - 1; i++) {
			appendBiasColumn(layers.get(i));
			preActivatedValue = matrixMultiplication(layers.get(i).layerValue, weightList.get(i));
			layers.get(i + 1).setLayerValue(preActivatedValue);
			layers.get(i + 1).preActivatedValue = preActivatedValue;
			layers.get(i + 1).setLayerValue(activate(layers.get(i + 1)));
		}
	}

	public void propagateInputLayer() {
		double[][] preActivatedValue;
		appendBiasColumn(layers.get(0));
		currentBatch = getBatch(layers.get(0));
		preActivatedValue = matrixMultiplication(currentBatch, weightList.get(0));
		layers.get(1).setLayerValue(preActivatedValue);
		layers.get(1).preActivatedValue = preActivatedValue;
		layers.get(1).setLayerValue(activate(layers.get(1)));
	}

	public void formatOutput() {
		System.out.println();
		System.out.println("Layer 0 " + java.util.Arrays.deepToString(currentBatch));
		for (int i = 1; i < layers.size(); i++) {
			System.out.println("Layer " + i + java.util.Arrays.deepToString(layers.get(i).layerValue));
		}
		
		System.out.println("Target Batch: " + java.util.Arrays.deepToString(currentTargetBatch));
		System.out.println();
		System.out.println("Loss: " + reportLoss(layers.get(layers.size() - 1))); // returns the final layerValue
		System.out.println();
		for (int i = 0; i < gradients.size(); i++) {
			System.out.println("Gradient " + java.util.Arrays.deepToString(gradients.get(i)));
		}
	}

	public double reportLoss(Layer finalLayer) {
		double loss = 0;
		double[][] result = copyArray(finalLayer.layerValue);
		double[][] targets = copyArray(currentTargetBatch);

		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {
					loss += ((targets[i][j] * Math.log(result[i][j]))
							+ ((1.0 - targets[i][j]) * Math.log(1.0 - result[i][j])));
				}
			}
			loss *= (-1.0 / (double) result.length);
		} else {
			for (int i = 0; i < result.length; i++) {
				loss += Math.pow((targets[i][0] - result[i][0]), 2);
			}
			loss *= ((1 / (double) result.length) * (1.0 / 2.0));
		}
		return loss;
	}

	public void appendBiasColumn(Layer layer) {
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

		layer.setLayerValue(inputsWithBiases);
	}

	int batchTracker = 0;
	int batchCounter = 0;
	int remainingBatchSize = 0;

	public double[][] getBatch(Layer layer) {
		double[][] batch;
		remainingBatchSize = (layer.layerValue.length % batchSize);

		if (remainingBatchSize == 0) {
			remainingBatchSize = batchSize;
		}

		if (!hasReachedEndofBatch()) {
			batch = new double[batchSize][layer.layerSize + 1];
			System.out.println("Ran batch");

			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < layer.layerSize + 1; j++) { // added one for the bias column
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++;
			}
			batchCounter++;

		} else {
			batch = new double[remainingBatchSize][layer.layerSize + 1];
			System.out.println("Ran remaining batch"); 
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
		if (calculateNumofBatches() - 1 == batchCounter) {
			return true;
		} else {
			return false;
		}
	}
	public boolean hasReachedEndofBatchTarget() {
		if (calculateNumofBatches() - 1 == batchCounterTarget) {
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
			System.out.println("Ran non remaining target batch " + batchSize);

			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					batch[i][j] = targets.targets[batchTrackerTarget][j];
				}
				batchTrackerTarget++; 
			}
			batchCounterTarget++; 
			currentTargetBatch = batch; 
		} else {
			System.out.println("Ran remaining target batch");
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

	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer);
		return activatedValue;
	}

	public void backPropagation() {
		computeGradients(); // remember to delete gradients from the list when done
		updateBiasedFirstMomentEstimate();
		updateBiasedSecondMomentEstimate();
		computeBiasCorrectedFirstMoment();
		computeBiasCorrectedSecondMoment();
		updateParameters();
	}

	double[][] previousPartialGradient;
	List<double[][]> gradients = new ArrayList<double[][]>();

	private void computeGradients() {
		double[][] gradient;
		
		previousPartialGradient = computePartialGradientLastLayer(layers.get(layers.size() - 1));
		gradient = matrixMultiplication(matrixTranspose(layers.get(layers.size() - 2).layerValue),
				previousPartialGradient);
		gradients.add(gradient);

		for (int i = weightList.size() - 1; i > 0; i--) { // must be edited because hidden layers are propagated
			gradient = matrixMultiplication(previousPartialGradient, matrixTranspose(weightList.get(i)));
			layers.get(i).preActivatedValue = concatenateColumn(layers.get(i).preActivatedValue);
			gradient = elementwiseMultiplication(gradient, computeDerivative(layers.get(i)));
			gradient = removeBiasColumn(gradient);
			previousPartialGradient = gradient;
			if (i != 1) {
				gradient = matrixMultiplication(matrixTranspose(layers.get(i - 1).layerValue), gradient);
			} else {
				gradient = matrixMultiplication(matrixTranspose(currentBatch), gradient);
			}
			gradients.add(gradient);
		}
	}

	private double[][] concatenateColumn(double[][] input) {
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

	private double[][] computeDerivative(Layer input) {
		double[][] derivative;
		derivative = activator.computeActivatedDerivative(input);
		return derivative;
	}

	private double[][] removeBiasColumn(double[][] layerValue) {
		double[][] result = new double[layerValue.length][layerValue[0].length - 1];
		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length - 1; j++) {
				result[i][j] = layerValue[i][j];
			}
		}
		return result;
	}

	private double[][] computePartialGradientLastLayer(Layer finalLayer) {
		double[][] partialGradient;
		partialGradient = elementwiseMultiplication((computeDerivativeofError(getTargetBatch(), finalLayer)),
				computeDerivative(finalLayer)); // might have to change part of this to negative
		return partialGradient;
	}

	private double[][] computeDerivativeofError(double[][] targetBatch, Layer finalLayer) {
		double[][] derivativeOfError = new double[targetBatch.length][targetBatch[0].length];
		double[][] finalLayerValue = copyArray(finalLayer.layerValue);

		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < targetBatch.length; i++) {
				for (int j = 0; j < targetBatch[0].length; j++) {
					derivativeOfError[i][j] = -1 * (targetBatch[i][j] * (1 / (double) finalLayerValue[i][j]))
							+ ((1.0 - targetBatch[i][j]) * (1.0 / (double) (1.0 - finalLayerValue[i][j])));
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

	private double[][] elementwiseMultiplication(double[][] m, double[][] n) {
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

	private double[][] matrixTranspose(double[][] m) {
		double[][] temp = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	private void updateBiasedFirstMomentEstimate() {
		// TODO Auto-generated method stub

	}

	private void updateBiasedSecondMomentEstimate() {
		// TODO Auto-generated method stub

	}

	private void computeBiasCorrectedFirstMoment() {
		// TODO Auto-generated method stub

	}

	private void computeBiasCorrectedSecondMoment() {
		// TODO Auto-generated method stub

	}

	private void updateParameters() {
		// TODO Auto-generated method stub

	}

	private double[][] matrixMultiplication(double[][] A, double[][] B) {

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
		for(int i=0; i<input.length; i++) {
			for(int j=0; j<input[0].length; j++) {
				copy[i][j] = input[i][j];
			}
		}
		return copy;
	}
}
