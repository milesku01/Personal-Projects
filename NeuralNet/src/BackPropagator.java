import java.util.ArrayList;
import java.util.List;

public class BackPropagator {
	final double regularize = 0.001;
	int listSize = 0; 
	static int batchSize;
	static int remainingBatchSize;
	static double[][] gradient;
	
	List<BackPropagator> propagationObjects = new ArrayList<BackPropagator>();
	
	static List<double[][]> weightList;
	static List<double[][]> weightChanges;

	static double[][] previousPartialGradient;

	InputLayer inputLayer;
	Layer finalLayer; 
	static Targets targets;
	static Gradients gradients;
	List<double[][]> gradientList  = new ArrayList<double[][]>(); 
	Activator activator = new Activator();
	Optimizer optimizer = new Optimizer(); 
	
	static List<Layer> layerList = new ArrayList<Layer>();
	
	
	
	public void runBackPropagation(String optimizerString) {
		
		for (int i = layerList.size() - 1; i > 0; i--) {
			gradientList.add(propagationObjects.get(layerList.size() - 1 - i).computeGradients(layerList.get(i), layerList.get(i - 1)));

		}
		
		weightChanges = optimizer.optimize(gradientList, optimizerString);
		updateParameters();
	}
	
	public double[][] computeGradients(Layer layer, Layer nextLayer) {
		
		return propagationObjects.get(listSize - layer.layerPosition).computeGradients(layer, nextLayer);

	}

	int weightListCounter = 0;
	private void updateParameters() {

		for (int i = 0; i < weightChanges.size(); i++) {
			if (weightChanges.get(i) instanceof double[][]) {
				for (int j = 0; j < ((double[][]) weightChanges.get(i)).length; j++) {
					for (int k = 0; k < ((double[][]) weightChanges.get(i))[0].length; k++) {
						weightList.get(weightListCounter)[j][k] -= weightChanges.get(i)[j][k];
					}
				}
				weightListCounter++;
			}
		}
		weightListCounter = 0;
	}
	
	
	public void constructBackwardPropagationObjects(NetworkModel model, Weights weights) { // only occur once
		BackPropagator backPropObj;
		setupConstants(model, weights);

		for (int i = layerList.size() - 1; i >= 0; i--) { // minus one because returns last value
			if (layerList.get(i) instanceof HiddenLayer || layerList.get(i) instanceof DropoutLayer) {
				backPropObj = new DenseBackPropagator();
				propagationObjects.add(backPropObj);
			} else if (layerList.get(i) instanceof OutputLayer) {
				backPropObj = new OutputBackPropagator();
				propagationObjects.add(backPropObj);
			}
		}

	}

	private void setupConstants(NetworkModel model, Weights weights) {
		layerList = model.layerList;
		listSize = layerList.size(); 
		inputLayer = (InputLayer) layerList.get(0);
		finalLayer = layerList.get(layerList.size() - 1); 
		targets = Layer.targets; //TEMP FIX 
		batchSize = Layer.batchSize;
		remainingBatchSize = Layer.remainingBatchSize;
		weightList = weights.weightList;
	}

	protected double[][] computeDerivative(Layer input) {
		return activator.computeActivatedDerivative(input);
	}
	
	protected void regularize(double[][] gradients, Layer layer) {
		for (int j = 0; j < gradient.length; j++) {
			for (int k = 0; k < gradient[0].length; k++) {
				gradient[j][k] += regularize * weightList.get(layer.layerPosition - 1)[j][k]; //check minus one 
			} 
		}
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

	public double[][] computeGradients(Layer layer, Layer nextLayer) {
		
		gradient = Utility.matrixMultiplication(previousPartialGradient, Utility.matrixTranspose(weightList.get(layer.layerPosition)));
		layer.preActivatedValue = concatenateColumn(layer.preActivatedValue);
		gradient = Utility.elementwiseMultiplication(gradient, computeDerivative(layer));
		gradient = removeBiasColumn(gradient);
		previousPartialGradient = gradient;
		
		if ((nextLayer instanceof HiddenLayer) || (nextLayer instanceof DropoutLayer)) {
			gradient = Utility.matrixMultiplication(Utility.matrixTranspose(nextLayer.layerValue), gradient);
		} else { //nextLayer is the inputLayer  
			gradient = Utility.matrixMultiplication(Utility.matrixTranspose(((InputLayer) nextLayer).currentBatch), gradient);
		}

		if (layer.layerPosition == 1) { //reached final two layers 
			for (int j = 0; j < gradient.length; j++) {
				for (int k = 0; k < gradient[0].length; k++) {
					gradient[j][k] *= ((double) remainingBatchSize / (double) batchSize);
				}
			}
		}

		regularize(gradient, layer);


		return gradient;
	}

}

class OutputBackPropagator extends BackPropagator {
	
	int targetBatchTracker = 0;

	private double[][] computePartialGradientLastLayer(OutputLayer finalLayer) {
		double[][] partialGradient;
		
		partialGradient = Utility.elementwiseMultiplication((computeDerivativeofError(getTargetBatch(finalLayer), finalLayer)),
					computeDerivative(finalLayer));
		
		if (!finalLayer.activation.equals("SOFTMAX"))  {
			Utility.scalarMultiply(-1.0, partialGradient);
		}
		
		return partialGradient;
	}

	private double[][] computeDerivativeofError(double[][] targetBatch, Layer finalLayer) {
		double[][] derivativeOfError = new double[targetBatch.length][targetBatch[0].length];
		
		if (finalLayer.activation.equals("SOFTMAX")) { //derivative of softmax error 
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

	public double[][] computeGradients(Layer layer, Layer nextLayer) {
		previousPartialGradient = computePartialGradientLastLayer((OutputLayer)layer);
		
	
		if (layerList.size() > 2) { // must add layer size handling, must have input and output layer (at least two
									// layers)
			gradient = Utility.matrixMultiplication(Utility.matrixTranspose(nextLayer.layerValue), previousPartialGradient);
		} else {
			System.out.println(nextLayer.currentBatch.length);
			gradient = Utility.matrixMultiplication(Utility.matrixTranspose(nextLayer.currentBatch), previousPartialGradient);
		}

		regularize(gradient, layer); 
	
		return gradient;
	}

	private double[][] getTargetBatch(OutputLayer layer) {
		double[][] batch = layer.batchList.get(targetBatchTracker).batchValue;
		
		layer.currentBatch.batchValue = batch; 

		if (layer.batchList.get(targetBatchTracker).finalBatch) {
			targetBatchTracker = 0;
		} else {
			targetBatchTracker++;
		}
		return batch;
	}
}
