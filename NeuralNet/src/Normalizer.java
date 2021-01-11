
public class Normalizer {

	double[] meanArray;
	double[] strdDev;
	
	double[] imageMean; 
	double[] imageStrdDev; 

	public void normalizeInputs(double[][] inputs, double[] mean, double[] strdDev) {
		inputs = jiggleInputs(inputs);
		for (int i = 0; i < inputs[0].length; i++) {
			for (int j = 0; j < inputs.length; j++) {
				inputs[j][i] = ((inputs[j][i] - mean[i]) / strdDev[i]);
			}
		}
	}

	public void normalizeInputsZscore(double[][] inputs, int targetSize) {
		calculateStrdDev(inputs);

		for (int i = 0; i < inputs[0].length - targetSize; i++) {
			if(meanArray[i] != 0.0 && strdDev[i] != 0.0) {
				for (int j = 0; j < inputs.length; j++) {
					inputs[j][i] = ((inputs[j][i] - meanArray[i]) / strdDev[i]);
				}
			}
		}
	}
	
	public double[][] normalizeInputsTanh(double[][] inputs, int targetSize) {
		calculateStrdDev(inputs);
		for (int i = 0; i < inputs[0].length - targetSize; i++) {
			if(meanArray[i] != 0.0 && strdDev[i] != 0.0) {
				for (int j = 0; j < inputs.length; j++) {
					inputs[j][i] = .5*(Math.tanh(.01*((inputs[j][i] - meanArray[i])/(strdDev[i]))) + 1);
				}
			}
		}
		return inputs;
	}
	
	public double[][] normalizeInputsMinMax(double[][] inputs, int targetSize) {
		double[] min = getMin(inputs); 
		double[] max = getMax(inputs); 
		
		for (int i = 0; i < inputs[0].length - targetSize; i++) {
			if(min[i]<max[i]) {
				for (int j = 0; j < inputs.length; j++) {
					inputs[j][i] = (inputs[j][i]-((max[i]+min[i])/2)) / (double)((max[i]-min[i])/2);
				}
			}
			
		}
		
		return inputs; 
	}
	
	private double[] getMin (double[][] inputs) {
		double[] minArray = new double[inputs[0].length]; 
		double min = 10000000; 
		
		for (int i = 0; i < inputs[0].length; i++) {
			for(int j=0; j<inputs.length; j++) {
				if(inputs[j][i] < min) {
					minArray[i] = inputs[j][i];
				}
			}
			min = 10000000;
		}
		return minArray; 
		
	}
	
	private double[] getMax (double[][] inputs) {
		double[] maxArray = new double[inputs[0].length]; 
		double max = -10000000; 
		
		for (int i = 0; i < inputs[0].length; i++) {
			for(int j=0; j<inputs.length; j++) {
				if(inputs[j][i] > max) {
					maxArray[i] = inputs[j][i];
				}
			}
			max = -10000000;
		}
		return maxArray; 
		
	}

	public void calculateStrdDev(double[][] inputs) {
		calculateMean(inputs);
		strdDev = new double[inputs[0].length];
		inputs = jiggleInputs(inputs);

		for (int i = 0; i < inputs[0].length; i++) {
			for (int j = 0; j < inputs.length; j++) {
				strdDev[i] += Math.pow(inputs[j][i] - meanArray[i], 2); //changed
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