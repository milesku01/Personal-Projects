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
		
		gradients = propagationObjects.get(objectTracker).computeGradients(layer, nextLayer);

		if (objectTracker == (propagationObjects.size() - 1)) {
			objectTracker = 0;
			layerCounter = weightList.size(); // once propagated over output, will decrease by one
		} else {
			objectTracker++;
			layerCounter--;
		}

		return gradients;
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
			} else if(layerList.get(i) instanceof DropoutLayer) {
				backPropObj = new DenseBackPropagator();
				propagationObjects.add(backPropObj);
			}
		}

	}

	private void setupConstants(NetworkModel model, Weights weights) {
		BackPropagator.layerList = model.layerList;
		targets = ((InputLayer)model.layerList.get(0)).targets; //TEMP FIX 

			inputLayer = (InputLayer) layerList.get(0);
			batchSize = inputLayer.batchSize;
			remainingBatchSize = inputLayer.remainingBatchSize;
		

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
		
		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < targetBatch.length; i++) {
				for (int j = 0; j < targetBatch[0].length; j++) {

					if (finalLayer.layerValue[i][j] < 0.000000001) {
						finalLayer.layerValue[i][j] += .000000001;
					}
					if (finalLayer.layerValue[i][j] > .999999999) {
						finalLayer.layerValue[i][j] -= .000000001;
					}

					derivativeOfError[i][j] = -1 * (targetBatch[i][j] * (1 / finalLayer.layerValue[i][j]))
							+ ((1.0 - targetBatch[i][j]) * (1.0 / (1.0 - finalLayer.layerValue[i][j])));
				}
			}

		} else {
			for (int i = 0; i < targetBatch.length; i++) {
				for (int j = 0; j < targetBatch[0].length; j++) {
					derivativeOfError[i][j] = targetBatch[i][j] - finalLayer.layerValue[i][j];
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
		if (InputLayer.numofBatches - 1 == batchCounterTarget) {
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
