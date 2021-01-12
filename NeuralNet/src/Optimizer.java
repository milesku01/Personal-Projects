import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Optimizer {
	Optimizer optimizationObject;
	List<double[][]> weightChange;
	public boolean TorF = true;
	// public double learningRate = .000008;
	public double learningRate = .01;

	private void createOptimizerObject(String optimizerString) {
		if (optimizerString.equals("ADAM")) {
			optimizationObject = new Adam();
		} else if (optimizerString.equals("BASIC")) {
			optimizationObject = new Basic();
		} else if (optimizerString.equals("MOMENTUM")) {
			optimizationObject = new Momentum();
		} else {
			optimizationObject = new Basic(); // defaults to basic
		}

	}

	public List<double[][]> optimize(List<double[][]> gradientList, String optimizerString) {
		if (TorF) {
			createOptimizerObject(optimizerString);
			TorF = false;
		}
		weightChange = optimizationObject.optimize(gradientList, "");
		return weightChange;
	}

	protected List<double[][]> initializeList(List<double[][]> gradients) { // TODO: fix this
		double[][] twoArray;
		List<double[][]> newList = new ArrayList<double[][]>(gradients.size());

		for (int i = 0; i < gradients.size(); i++) {
			if (gradients.get(i) != null) {
				twoArray = new double[gradients.get(i).length][gradients.get(i)[0].length];
				newList.add(twoArray);
			}
		}

		return newList;
	}
}

class Adam extends Optimizer {

	double beta1 = .9;
	double beta2 = .999;
	final double offSet = .000000001;
	int betaCounter = 1;

	List<double[][]> firstMomentEstimate;
	List<double[][]> secondMomentEstimate;
	List<double[][]> firstMomentEstimateCorrected;
	List<double[][]> secondMomentEstimateCorrected;
	List<double[][]> gradientCopy;

	public List<double[][]> optimize(List<double[][]> gradients, String optimizerString) {
		Collections.reverse(gradients);
		gradientCopy = gradients; // for public access

		if (TorF == true) {
			weightChange = initializeList(gradients);
			initializeMomentLists(gradients);
			TorF = false;
		}

		updateBiasedFirstMomentEstimate();
		updateBiasedSecondMomentEstimate();
		computeBiasCorrectedFirstMoment();
		computeBiasCorrectedSecondMoment();

		calculateParameterUpdate();

		// gradientCopy.clear();
		gradients.clear();

		return weightChange;
	}

	private void initializeMomentLists(List<double[][]> gradients) {
		firstMomentEstimate = initializeList(gradients);
		secondMomentEstimate = initializeList(gradients);
		firstMomentEstimateCorrected = initializeList(gradients);
		secondMomentEstimateCorrected = initializeList(gradients);
	}

	private void updateBiasedFirstMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(firstMomentEstimate.get(k))[i][j] = (beta1 * (firstMomentEstimate.get(k))[i][j])
								+ ((1.0 - beta1) * (gradientCopy.get(k)[i][j]));
					}
				}

			}
		}
	}

	private void updateBiasedSecondMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(secondMomentEstimate.get(k))[i][j] = (beta2 * (secondMomentEstimate.get(k))[i][j])
								+ ((1.0 - beta2) * gradientCopy.get(k)[i][j] * gradientCopy.get(k)[i][j]);
					}
				}
			}
		}

	}

	private void computeBiasCorrectedFirstMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(firstMomentEstimateCorrected.get(k))[i][j] = (firstMomentEstimate.get(k))[i][j]
								/ (1.0 - Math.pow(beta1, betaCounter));
					}
				}

			}

		}
	}

	private void computeBiasCorrectedSecondMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(secondMomentEstimateCorrected.get(k))[i][j] = (secondMomentEstimate.get(k))[i][j]
								/ (1.0 - Math.pow(beta2, betaCounter));
					}
				}

			}
		}
	}

	private void calculateParameterUpdate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(weightChange.get(k))[i][j] = ((learningRate * (firstMomentEstimateCorrected.get(k))[i][j])
								/ (Math.sqrt((secondMomentEstimateCorrected.get(k))[i][j]) + offSet));
					}
				}
			}
		}

		betaCounter++;
	}

}

class Basic extends Optimizer {

	List<double[][]> gradientCopy;

	public List<double[][]> optimize(List<double[][]> gradients, String optimizerString) {
		Collections.reverse(gradients);
		gradientCopy = gradients; // for public access

		if (TorF == true) {
			weightChange = initializeList(gradients);
			TorF = false;
		}
		calculateParameterUpdate();
		return weightChange;
	}

	private void calculateParameterUpdate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(weightChange.get(k))[i][j] = learningRate * (gradientCopy.get(k)[i][j]);
					}
				}

			}
		}
	}

}

class Momentum extends Optimizer {

	double beta = .9;
	List<double[][]> gradientCopy;

	public List<double[][]> optimize(List<double[][]> gradients, String optimizerString) {
		Collections.reverse(gradients);
		gradientCopy = gradients; // for public access

		if (TorF == true) {
			weightChange = initializeList(gradients);
			TorF = false;
		}

		calculateParameterUpdate();

		gradients.clear();

		return weightChange;
	}

	private void calculateParameterUpdate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k) != null) {
				for (int i = 0; i < gradientCopy.get(k).length; i++) {
					for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
						(weightChange.get(k))[i][j] = beta * (weightChange.get(k))[i][j]
								+ (learningRate) * (gradientCopy.get(k)[i][j]);
					}
				}

			}
		}
	}

}
