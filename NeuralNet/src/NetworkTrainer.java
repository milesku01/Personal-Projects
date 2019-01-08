import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;

public class NetworkTrainer {
	int batchSize = 0;
	int numofEpochs = 0;
	static int remainingBatchSize;
	List<Layer> layers;
	
	InputLayer inputLayer;
	Targets targets;
	List<double[][]> weightList;
	Activator activator = new Activator();
	Optimizer optimizer = new Optimizer();
	ForwardPropagator fp = new ForwardPropagator(); 
	BackPropagator bp = new BackPropagator(); 
	Layer layer = new Layer();
	double[][] fullFinalLayer;
	static int numofBatches;
	String optimizerString;
	String[] activatorStrings;

	public void train(NetworkModel model, Weights weights, int numofEpochs, String optimizerString) {
		this.numofEpochs = numofEpochs;
		layers = model.layerList;
		activatorStrings = new String[layers.size() - 1];
		inputLayer = (InputLayer) layers.get(0);
		batchSize = inputLayer.batchSize;
		weightList = weights.weightList;
		targets = model.targets;
		this.optimizerString = optimizerString;

		if((layers.get(0).layerValue.length % batchSize) == 0) {
			remainingBatchSize = batchSize; 
		} else {
			remainingBatchSize = (layers.get(0).layerValue.length % batchSize);
		}
		
		getActivatorStrings();
		numofBatches = calculateNumofBatches();
		
		
		int iterations = numofBatches * numofEpochs;
		//int iterations = 1; 
		
		fullFinalLayer = new double[targets.targets.length][targets.targets[0].length];
		
		fp.constructForwardPropagationObjects(layers, weightList);
		bp.constructBackwardPropagationObjects(model, weightList); 

		long startTime = System.nanoTime();

		for (int i = 1; i <= iterations + 1; i++) {
			forwardPropagation();
			if (i != iterations + 1)backPropagation();
			formatOutput(i);
		}

		long endTime = System.nanoTime();
		System.out.println(" \nTraining time: " + getTrainingTime(startTime, endTime) + " sec");
	}

	private void getActivatorStrings() {
		for (int i = 0; i < activatorStrings.length; i++) {
			activatorStrings[i] = layers.get(i + 1).activation;
		}
	}

	private String getTrainingTime(long startTime, long endTime) {
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

	public void forwardPropagation() {
		
		for(int i = 0; i < layers.size() - 1; i++) {
			layers.get(i + 1).layerValue = fp.propagate(layers.get(i), layers.get(i + 1)); //nextLayer, previousLayer
		}
		
	}

	int targetPositionCounter = 0;

	private void determineTargetForError() {
		
		if (targetPositionCounter != (batchSize * (numofBatches - 1))) {
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					fullFinalLayer[i + targetPositionCounter][j] = layers.get(layers.size() - 1).layerValue[i][j];
				}
			}
			targetPositionCounter += batchSize;
		} else {
			for (int i = 0; i < remainingBatchSize; i++) {
				for (int j = 0; j < targets.targetSize; j++) {
					fullFinalLayer[i + targetPositionCounter][j] = layers.get(layers.size() - 1).layerValue[i][j];
				}
			}
			targetPositionCounter = 0;
		}
	
	}

	double[][] finalTestLayer;
	double[][] layerValue;

	private void determineAccuracy() {
		finalTestLayer = forwardPropTest();
		if(activatorStrings[activatorStrings.length - 1].equals("SOFTMAX")) {
			System.out.println("Accuracy " + computeAccuracy());
		} else {
			System.out.println("Loss over test set " + computeAccuracy()); 
		}
	}

	public double[][] forwardPropTest() {
		for(int i = 0; i < layers.size() - 1; i++) {
			layers.get(i + 1).layerValue = fp.propagateTest(layers.get(i), layers.get(i + 1)); //nextLayer, previousLayer
		}
	
		return layers.get(layers.size()-1).layerValue;
	}

	

	public void formatOutput(int i) {
		if (i % numofBatches == 0) {
			System.out.println();

			if (inputLayer.numofSets > 140) {
				determineAccuracy();
			}
			
			
			 
		//	System.out.println("Current batch " + java.util.Arrays.deepToString(layers.get(0).currentBatch));
		//	 System.out.println("Layer 0 " + java.util.Arrays.deepToString(layers.get(0).layerValue));
		//	  for (int j = 1; j < layers.size() - 1; j++) { System.out.println("Layer " +
		//	  java.util.Arrays.deepToString(layers.get(j).layerValue)); }
		//	  System.out.println("Last Layer " +
		//	  java.util.Arrays.deepToString(fullFinalLayer));
			  
		//	  System.out.println("Targets: " +
		//	  java.util.Arrays.deepToString(targets.targets));
			  //System.out.println(); 
		//	  for (int j = 0; j < weightList.size(); j++) { 
		//		  System.out.println("weight " + j + java.util.Arrays.deepToString(weightList.get(j))); 
		//	  }
		 
		 
			System.out.println("Loss: " + reportLoss(layers.get(layers.size() - 1))); // returns the final layerValue
		
		}
	}

	private double computeAccuracy() {
		int correct = 0; 
		double loss = 0;
		double[][] result = copyArray(finalTestLayer);
		double[][] target = copyArray(targets.testTargets);
		String lastAct = activatorStrings[activatorStrings.length - 1];

		if (lastAct.equals("SOFTMAX")) {
			for(int i=0; i<result.length; i++) {
				if(Math.abs(target[i][0] - result[i][0]) < .5) {
					correct++; 
				}
			}
		} else {
			
			for (int i = 0; i < result.length; i++) {
				loss += Math.pow((target[i][0] - result[i][0]), 2);
			}
			loss *= ((1 / (double) result.length) * (1.0 / 2.0));
		}

		if (lastAct.equals("SOFTMAX")) {
			return (double)(correct/(double)result.length); // must change
		} else {
			return loss;
		}

	}

	public double reportLoss(Layer finalLayer) {
		double loss = 0;
		double[][] result = copyArray(fullFinalLayer);
		double[][] target = copyArray(targets.targets);
	
		if (finalLayer.activation.equals("SOFTMAX")) {
			for (int i = 0; i < result.length; i++) {
				for (int j = 0; j < result[0].length; j++) {
						
					if(result[i][j] == 0.0) {
						result[i][j] += .000000001;
					} else if(result[i][j] == 1.0){
						result[i][j] -= .000000001;
					}
				
					loss += ((target[i][j] * Math.log(result[i][j]))
							+ ((1.0 - target[i][j]) * Math.log(1.0 - result[i][j])));
				}
			}
			loss *= (-1.0 / (double) result.length);
		} else {
			for (int i = 0; i < result.length; i++) {
				loss += Math.pow((target[i][0] - result[i][0]), 2);
			}
			loss *= ((1 / (double) result.length) * (1.0 / 2.0));
		}
		System.out.println("Reg " + regularizationTerm());
		loss += regularizationTerm();
		return loss;
	}

	private double regularizationTerm() {
		double weightSquaredSum = 0.0;
		for (int i = 0; i < weightList.size(); i++) {
			for (int j = 0; j < weightList.get(i).length; j++) {
				for (int k = 0; k < weightList.get(i)[0].length; k++) {
					weightSquaredSum += Math.pow(weightList.get(i)[j][k], 2);
				}
			}
		}
		weightSquaredSum *= bp.regularize / (2.0 * (double) inputLayer.currentBatch.length);
		return weightSquaredSum;
	}

	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer);
		return activatedValue;
	}

	List<double[][]> weightChanges;

	public void backPropagation() {
		determineTargetForError(); 
		computeGradients();
		weightChanges = optimizer.optimize(gradients, optimizerString);
		updateParameters();
		cleanGradients();
	}
	
	private void computeGradients() {
		double[][] gradient = null; 
		for(int i = layers.size()-1; i > 0; i--) {
			gradient = bp.computeGradients(layers.get(i), layers.get(i-1)); 
			gradients.add(gradient); 
		}
	}
	
	List<double[][]> gradients = new ArrayList<double[][]>();
	
	private void cleanGradients() {
		gradients.clear();
	}


	private void updateParameters() {
		for (int i = 0; i < weightChanges.size(); i++) {
			for (int j = 0; j < weightChanges.get(i).length; j++) {
				for (int k = 0; k < weightChanges.get(i)[0].length; k++) {
					weightList.get(i)[j][k] -= weightChanges.get(i)[j][k];
				}
			}
		}
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
