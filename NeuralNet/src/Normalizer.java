 public class Normalizer {

	double[] meanArray;
	double[] strdDev;

	public double[][] normalizeInputs(double[][] inputs, double[] mean, double[] strdDev) {
		inputs = jiggleInputs(inputs);
		for (int i = 0; i < inputs[0].length; i++) {
			for (int j = 0; j < inputs.length; j++) {
				inputs[j][i] = ((inputs[j][i] - mean[i]) / strdDev[i]);
			}
		}
		return inputs;
	}

	public double[][] normalizeInputs(double[][] inputs, int targetSize) {
		calculateStrdDev(inputs);
		for (int i = 0; i < inputs[0].length - targetSize; i++) {
			for (int j = 0; j < inputs.length; j++) {
				inputs[j][i] = ((inputs[j][i] - meanArray[i]) / strdDev[i]);
			}
		}
		return inputs;
	}

	public void calculateStrdDev(double[][] inputs) {
		calculateMean(inputs);
		strdDev = new double[inputs[0].length];
		inputs = jiggleInputs(inputs);

		for (int i = 0; i < inputs[0].length; i++) {
			for (int j = 0; j < inputs.length; j++) {
				strdDev[i] = Math.pow(inputs[j][i] - meanArray[i], 2);
			}
		}
		for (int i = 0; i < inputs[0].length; i++) {
			strdDev[i] = strdDev[i] / (double) inputs.length;
		}
		for (int i = 0; i < inputs[0].length; i++) {
			strdDev[i] = Math.sqrt(strdDev[i]);
		}
	}

	public void calculateMean(double[][] inputs) {

		meanArray = new double[inputs[0].length];
		double runningTotal = 0.0;

		for (int i = 0; i < inputs[0].length; i++) {
			for (int j = 0; j < inputs.length; j++) {
				runningTotal += inputs[j][i];
			}
			meanArray[i] = (runningTotal / (double) inputs.length);
			runningTotal = 0.0;
		}
	}

	public double[][] jiggleInputs(double[][] inputs) {
		for (int i = 0; i < inputs[0].length; i++) {
			inputs[0][i] = inputs[0][i] + .0000000001;
		}
		return inputs;
	}

}