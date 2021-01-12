import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.List;

public class NetworkTrainer {

	static int numofLayers;

	static int remainingBatchSize;
	List<Layer> layers;

	double[][] finalTestLayer;

	InputLayer inputLayer;

	Targets targets;

	ForwardPropagator fp = new ForwardPropagator();
	BackPropagator bp = new BackPropagator();

	Weights weights;

	double accuracy = 0;
	String optimizerString;
	
	

	public void train(NetworkModel model, int numofEpochs, String optimizerString) {
		layers = model.layerList;
		numofLayers = layers.size();
		inputLayer = (InputLayer) layers.get(0);

		weights = new Weights();

		weights.generateInitialWeights(model);

		this.optimizerString = optimizerString;

		int iterations = InputLayer.numofBatches * numofEpochs;

		fp.constructForwardPropagationObjects(model, weights);
		bp.constructBackwardPropagationObjects(model, weights);

		long startTime = System.nanoTime();

		for (int i = 1; i <= iterations + 1; i++) {

			forwardPropagation();

			if (i != iterations + 1) {
				backPropagation(); 
				formatOutput(i); //keep in mind this means the loss is not reported for the last entry 
								// this is because if the when the back propagation is done for the last time, the 
								//target batch changes while the results do not, i.e.
			}
			
			
		}

		if (Utility.testSplitThreshold(layers.get(0))) {
			determineAccuracy();
		}

		long endTime = System.nanoTime();
		System.out.println(" \nTraining time: " + getTrainingTime(startTime, endTime) + " sec");
	}

	
	public void trainUntil(NetworkModel model, double accuracy, int numofEpochs, String optimizerString) {
		
		
		if (model.layerList.get(model.layerList.size()-1).activation.equals("SOFTMAX")) {
			while (this.accuracy < accuracy) {
				train(model, numofEpochs, optimizerString);
			}
		} else {
			do {
				train(model, numofEpochs, optimizerString);
			} while (this.accuracy < accuracy);
		}
	}

	public String getTrainingTime(long startTime, long endTime) {
		Double n = (double) (endTime - startTime) / (double) 1000000000;
		DecimalFormat df = new DecimalFormat("#.####");
		df.setRoundingMode(RoundingMode.CEILING);
		Double d = n.doubleValue();
		return df.format(d);
	}

	public void printArray(double[][] d) {
		for (int i = 0; i < d.length; i++) {
			for (int j = 0; j < d[0].length; j++) {
				System.out.print(Math.round(d[i][j] * 10000) / 10000.0 + " ");
			}
		}
		System.out.println();
	}

	public void forwardPropagation() {

		fp.runPropagation();

	}

	private void determineAccuracy() {
		finalTestLayer = forwardPropTest();
		if (layers.get(layers.size() - 1).activation.equals("SOFTMAX")) {
			System.out.println("Accuracy " + computeAccuracy());
		} else {
			System.out.println("Loss over test set " + computeAccuracy());
		}
	}

	public double[][] forwardPropTest() {

		fp.runPropagationTest();

		return layers.get(layers.size() - 1).testData;
	}

	public void formatOutput(int i) {
		// if (i % numofBatches == 0) {
		// System.out.println();

		// System.out.println("Current batch " +
		// java.util.Arrays.deepToString(layers.get(0).currentBatch));

		// for (int j = 0; j < layers.size() - 1; j++) {
		// System.out.println(layers.get(i).layerValue.length + " " +
		// layers.get(i).layerValue[0].length);
		// }

		// for(int i1=0; i1 < layers.size(); i1++) {
		// System.out.println(java.util.Arrays.deepToString(layers.get(i1).preActivatedValue));

		// System.out.println(java.util.Arrays.deepToString(layers.get(i1).layerValue));

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

	//	System.out.println("Loss: " + reportLoss(layers.get(layers.size() - 1))); //
	//	System.out.println("Regularization: " + regularizationTerm());
		// }
	}

	private double computeAccuracy() {
		int correct = 0;
		double loss = 0;
		double[][] result = finalTestLayer; 
		double[][] target = BackPropagator.targets.testTargets;
		String lastAct = layers.get(layers.size() - 1).activation;

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
			loss *= (.5 / (result.length * result[0].length));

			if (result[0].length == 2) {
				for (int i = 0; i < result.length; i++) {
					if (((result[i][0] > result[i][1]) && (target[i][0] > target[i][1]))
							|| ((result[i][0] < result[i][1]) && (target[i][0] < target[i][1]))) {
						correct++;
					}
				}
				System.out.println("Accuracy " + (correct / (double) result.length));
			}
		}

		if (lastAct.equals("SOFTMAX")) {
			accuracy = (correct / (double) result.length);
			return accuracy; // must change
		}

		else {
			accuracy = (correct / (double) result.length);
			return loss;
		}

	}

	public double reportLoss(Layer finalLayer) {
		double loss = 0;
		
		OutputLayer layer = (OutputLayer) finalLayer; 
		
		double[][] result = (finalLayer.layerValue); // needed to be copied?
		double[][] target = layer.currentBatch.batchValue;

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
			loss *= (.5 / (result.length * result[0].length));
		}
		loss += regularizationTerm();
		return loss;
	}

	public double regularizationTerm() {
		double currentLength = 1;
		double weightSquaredSum = 0.0;
		List<double[][]> weightList = BackPropagator.weightList;

		currentLength = bp.inputLayer.currentBatch.length;

		for (int i = 0; i < weightList.size(); i++) {
			for (int j = 0; j < weightList.get(i).length; j++) {
				for (int k = 0; k < weightList.get(i)[0].length; k++) {
					weightSquaredSum += Math.pow(weightList.get(i)[j][k], 2);
				}
			}
		}
		weightSquaredSum *= bp.regularize / (2.0 * currentLength);
		return weightSquaredSum;
	}

	public void backPropagation() {
		bp.runBackPropagation(optimizerString);
	}

}
