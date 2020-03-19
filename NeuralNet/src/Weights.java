import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Weights {
	private double[][] weightArray;
	List<double[][]> weightList;
	List<Filters> filterList;
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
		filterList = new ArrayList<Filters>(model.filterCount);

		if (model.modelType.equals("DIAGNOSTIC")) {
			createStandardWeights(model);
		} else {
			for (int i = 0; i < layerList.size() - 1; i++) { // finishes before the output layer is multiplied

				if (layerList.get(i) instanceof InputLayer || layerList.get(i) instanceof HiddenLayer
						|| layerList.get(i) instanceof DropoutLayer) {
					weightList.add(produceWeightObject(layerList.get(i), layerList.get(nextLayer)));
				} else if (layerList.get(i) instanceof ConvolutionalLayer) {

					ConvolutionalLayer conv = (ConvolutionalLayer) layerList.get(i);
					Filters filter = new Filters(conv.numofFilters, conv.filterSize, conv.channelDepth);
					filterList.add(produceFilterValues(filter, model));

				} else if (layerList.get(i) instanceof HiddenConvolutionalLayer) {

					HiddenConvolutionalLayer hConv = (HiddenConvolutionalLayer) layerList.get(i);

					Filters filter = new Filters(hConv.numofFilters, hConv.filterSize, hConv.channelDepth);

					filterList.add(produceHiddenFilterValues(filter));

					hConv.filterList = filter.threeDFilterArray; // maybe fine passing reference
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

	private Filters produceFilterValues(Filters filter, NetworkModel model) {

		int w = ((ConvolutionalLayer) model.layerList.get(0)).imageWidth;
		int h = ((ConvolutionalLayer) model.layerList.get(0)).imageHeight;
		int d = ((ConvolutionalLayer) model.layerList.get(0)).channelDepth;
		double n = (d * w * h);

		List<double[][][]> filterValuesList = new ArrayList<double[][][]>(filter.numofFilters);
		double[][][] filterValues = null;

		for (int i = 0; i < filter.numofFilters; i++) {
			filterValues = new double[d][filter.filterSize][filter.filterSize];
			for (int j = 0; j < d; j++) {
				for (int k = 0; k < filter.filterSize; k++) {
					for (int l = 0; l < filter.filterSize; l++) {
						filterValues[j][k][l] = r.nextGaussian() * Math.sqrt(2.0 / n);
					}
				}
			}
			filterValuesList.add(filterValues);
		}
		filter.threeDFilterArray = filterValuesList;

		return filter;
	}

	private Filters produceHiddenFilterValues(Filters filter) {

		double n = filter.filterSize * filter.filterSize * filter.previousDepth;

		System.out.println(filter.previousDepth);

		List<double[][][]> filterValuesList = new ArrayList<double[][][]>(filter.numofFilters);
		double[][][] filterValues = null;

		for (int i = 0; i < filter.numofFilters; i++) {
			filterValues = new double[filter.previousDepth][filter.filterSize][filter.filterSize];
			for (int j = 0; j < filter.previousDepth; j++) {
				for (int k = 0; k < filter.filterSize; k++) {
					for (int l = 0; l < filter.filterSize; l++) {
						filterValues[j][k][l] = r.nextGaussian() * Math.sqrt(2.0 / n); // between -2 and 2
					}
				}
			}
			filterValuesList.add(filterValues);
		}

		filter.threeDFilterArray = filterValuesList;

		return filter;
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
