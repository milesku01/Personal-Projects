import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Collections;
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
	
		long startTime = System.nanoTime(); 
		for (int i = 0; i < iterations; i++) {
			if(i == 0) initializeLists();
			forwardPropagation();
			backPropagation();
			formatOutput();
		}
		long endTime = System.nanoTime(); 
		System.out.println("Training time: " + getTrainingTime(startTime, endTime) + " sec"); 
	}

	private String getTrainingTime(long startTime, long endTime) {
		Double n = (double) (endTime - startTime)/(double)1000000000; 
		DecimalFormat df = new DecimalFormat("#.####");
		df.setRoundingMode(RoundingMode.CEILING);
		Double d = n.doubleValue();
	    return df.format(d);
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
		for (int i = 0; i < weightList.size(); i++) {
			System.out.println("weight " + i + java.util.Arrays.deepToString(weightList.get(i)));
		}
		System.out.println("Loss: " + reportLoss(layers.get(layers.size() - 1))); // returns the final layerValue
		System.out.println();
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
		loss += regularizationTerm(); 
		System.out.println("Reg. " + regularizationTerm());
		return loss;
	}
	private double regularizationTerm() {
		double weightSquaredSum = 0.0; 
		for(int i=0; i<weightList.size(); i++) { 
			for(int j=0; j < weightList.get(i).length; j++) {
				for(int k=0; k < weightList.get(i)[0].length; k++) {
					weightSquaredSum += Math.pow(weightList.get(i)[j][k], 2); 
				}
			}
		}
		weightSquaredSum *= regularize/(2.0*(double)currentBatch.length); 
		return weightSquaredSum; 
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
		formatLists(); 
		updateParameters();
		formatLists(); //reverse again 
		cleanGradients();
	}

	double[][] previousPartialGradient;
	List<double[][]> gradients = new ArrayList<double[][]>();
	double regularize = 0.001; 

	private void computeGradients() {
		double[][] gradient;
		
		previousPartialGradient = computePartialGradientLastLayer(layers.get(layers.size() - 1));
		gradient = matrixMultiplication(matrixTranspose(layers.get(layers.size() - 2).layerValue),
				previousPartialGradient);
		
		for(int i=0; i<gradient.length; i++) {
			for(int j=0; j<gradient[0].length; j++) {
				gradient[i][j] = gradient[i][j] + regularize*weightList.get(weightList.size()-1)[i][j];
			}
		}
		gradients.add(gradient);

		for (int i = weightList.size() - 1; i > 0; i--) { //no need for weightList.get(0) 
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
			
			for(int j=0; j<gradient.length; j++) {
				for(int k=0; k<gradient[0].length; k++) {
					gradient[j][k] = gradient[j][k] + regularize*weightList.get(i-1)[j][k];
				}
			}
			gradients.add(gradient);
		}
	}

	private void cleanGradients() {
		gradients.clear();
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
		if(finalLayer.activation.equals("SOFTMAX")) {
			partialGradient = elementwiseMultiplication((computeDerivativeofError(getTargetBatch(), finalLayer)),
				computeDerivative(finalLayer)); 
		} else {
			partialGradient = elementwiseMultiplication((computeDerivativeofError(getTargetBatch(), finalLayer)),
					scalarMultiply(-1.0, computeDerivative(finalLayer))); 
		}
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

	private double[][] scalarMultiply(double num, double[][] m) {
		for(int i=0; i<m.length; i++) {
			for(int j=0; j<m[0].length; j++) {
				m[i][j] = num*m[i][j];
			}
		}
		return m; 
	}
	
	double beta1 = .9;
	double beta2 = .999;
	double learningRate = .01; 
	double offSet = .000000001;
	int betaCounter = 1; 
	List<double[][]> firstMomentEstimate;
	List<double[][]> secondMomentEstimate;
	List<double[][]> firstMomentEstimateCorrected;
	List<double[][]> secondMomentEstimateCorrected;
 
	private void initializeLists() {
		firstMomentEstimate = initializeMomentList();
		secondMomentEstimate = initializeMomentList();
		firstMomentEstimateCorrected = initializeMomentList();
		secondMomentEstimateCorrected = initializeMomentList();
	}
	
	private List initializeMomentList() {
		double[][] array;
		List newList = new ArrayList(); 
		
		for(int i=weightList.size()-1; i>=0; i--) {
			array = new double[weightList.get(i).length][weightList.get(i)[0].length];
			newList.add(array);
		}
		return newList; 
	}

	private void updateBiasedFirstMomentEstimate() { 
		for(int k=0; k<gradients.size(); k++) {
			for(int i=0; i<gradients.get(k).length; i++) {
				for(int j=0; j<gradients.get(k)[0].length; j++) {
					firstMomentEstimate.get(k)[i][j] = (beta1*firstMomentEstimate.get(k)[i][j]) + 
							((1.0-beta1)* (gradients.get(k)[i][j]));
				}
			}
		}
	}
 
	private void updateBiasedSecondMomentEstimate() {
		for(int k=0; k<gradients.size(); k++) {
			for(int i=0; i<gradients.get(k).length; i++) {
				for(int j=0; j<gradients.get(k)[0].length; j++) {
					secondMomentEstimate.get(k)[i][j] = (beta2*secondMomentEstimate.get(k)[i][j]) + 
							((1.0-beta2)*gradients.get(k)[i][j]*gradients.get(k)[i][j]);
				}
			}
		}

	}

	private void computeBiasCorrectedFirstMoment() {
		for(int k=0; k<gradients.size(); k++) { 
			for(int i=0; i<gradients.get(k).length; i++) {
				for(int j=0; j<gradients.get(k)[0].length; j++) {
					firstMomentEstimateCorrected.get(k)[i][j] =	firstMomentEstimate.get(k)[i][j]/
							(1.0-Math.pow(beta1, betaCounter));
					
				}
			}
		}
	}

	private void computeBiasCorrectedSecondMoment() {
		for(int k=0; k<gradients.size(); k++) {
			for(int i=0; i<gradients.get(k).length; i++) {
				for(int j=0; j<gradients.get(k)[0].length; j++) {
					secondMomentEstimateCorrected.get(k)[i][j] = secondMomentEstimate.get(k)[i][j]/
							(1.0-Math.pow(beta2, betaCounter));
					
				}
			}
		}
	}

	private void formatLists() {
		Collections.reverse(firstMomentEstimateCorrected);
		Collections.reverse(secondMomentEstimateCorrected);
	}
	
	private void updateParameters() {
		for(int i=0; i < weightList.size(); i++) {
			for(int j=0; j <weightList.get(i).length; j++) {
				for(int k=0; k<weightList.get(i)[0].length; k++) {
					weightList.get(i)[j][k] -= ((((double) currentBatch.length/(double) batchSize)*learningRate*
							firstMomentEstimateCorrected.get(i)[j][k])/
								(Math.sqrt(secondMomentEstimateCorrected.get(i)[j][k])
										+ offSet));
				}
			}
		}
		betaCounter++; 
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
	
	private double[][] subtractAcross(double[][] A, double[][] B) {
		if (A.length != B.length || A[0].length != B[0].length) {
			throw new IllegalArgumentException(
					"Dimensions did not match" + A.length + " " + A[0].length + " " + B.length + " " + B[0].length);
		}
		for(int i=0; i<A.length; i++) {
			for(int j=0; j<A[0].length; j++) {
				A[i][j] = A[i][j] + B[i][j];
			}
		}
		return A; 
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
