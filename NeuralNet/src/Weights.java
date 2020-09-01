import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Weights {
	private double[][] weightArray;
	List<double[][]> weightList;
	Random r = new Random();

	public void createStandardWeights(NetworkModel model) {
		int nextLayer = 1;
		List<Layer> layerList = model.layerList;
		for (int i = 0; i < layerList.size() - 1; i++) { // finishes before the output layer is multiplied
			weightList.add(generateWeightStandard(layerList.get(i), layerList.get(nextLayer)));
			nextLayer++;
		}
	}

	private double[][] generateWeightStandard(Layer previousLayer, Layer nextLayer) {
		weightArray = new double[previousLayer.layerSize][nextLayer.layerSize];
		for (int i = 0; i < previousLayer.layerSize; i++) {
			for (int j = 0; j < nextLayer.layerSize; j++) {
				weightArray[i][j] = .2;
			}
		}
		return addWeightBiases(weightArray);
	}

	public void generateInitialWeights(NetworkModel model) {
		int nextLayer = 1;

		List<Layer> layerList = model.layerList;
		weightList = new ArrayList<double[][]>(model.weightListCount);

		if (model.modelType.equals("DIAGNOSTIC")) {
			createStandardWeights(model);
		} else {
			for (int i = 0; i < layerList.size() - 1; i++) { // finishes before the output layer is multiplied

				if (layerList.get(i) instanceof InputLayer || layerList.get(i) instanceof HiddenLayer
						|| layerList.get(i) instanceof DropoutLayer) {
					weightList.add(produceWeightObject(layerList.get(i), layerList.get(nextLayer)));
				} 

				nextLayer++;
			}
		}
	}

	public double[][] produceWeightObject(Layer previousLayer, Layer nextLayer) {
		weightArray = new double[previousLayer.layerSize][nextLayer.layerSize];
		for (int i = 0; i < previousLayer.layerSize; i++) {
			for (int j = 0; j < nextLayer.layerSize; j++) {
				weightArray[i][j] = r.nextGaussian() * Math.sqrt(2.0 / (double) (previousLayer.layerSize));
			}
		}
		return addWeightBiases(weightArray);
	}

	public double[][] addWeightBiases(double[][] weightValue) {
		double[][] weightsWithBiases = new double[weightValue.length + 1][weightValue[0].length];
		for (int i = 0; i < weightValue.length; i++) {
			for (int j = 0; j < weightValue[0].length; j++) {
				weightsWithBiases[i][j] = weightValue[i][j];
			}
		}
		for (int i = 0; i < weightValue[0].length; i++) {
			weightsWithBiases[weightValue.length][i] = .1;
		}
		return weightsWithBiases;
	}

}
