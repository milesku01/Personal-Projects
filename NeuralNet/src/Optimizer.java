import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Optimizer {
	Optimizer optimizationObject; 
	List<double[][]> weightChange; 
	public boolean TorF = true; 
	
	private void createOptimizerObject(String optimizerString) {
		if(optimizerString.equals("ADAM")) {
			optimizationObject = new Adam();
		} else if (optimizerString.equals("BASIC")) {
			optimizationObject =  new Basic();
		} else {
			optimizationObject =  new Basic(); //defaults to basic 
			
		}
		
	}
	
	public List<double[][]> optimize(List<double[][]> gradients, String optimizerString){
		if(TorF == true) {createOptimizerObject(optimizerString); TorF = false;}
		weightChange = optimizationObject.optimize(gradients, "");
		return weightChange; 
	}
}
class Adam extends Optimizer {
	 
	double beta1 = .9;
	double beta2 = .999;
	double learningRate = .01;
	final double offSet = .000000001;
	int betaCounter = 1;
	
	List<double[][]> firstMomentEstimate;
	List<double[][]> secondMomentEstimate;
	List<double[][]> firstMomentEstimateCorrected;
	List<double[][]> secondMomentEstimateCorrected;
	List<double[][]> gradientCopy;
	
	public List<double[][]> optimize(List<double[][]> gradients, String optimizerString){
		gradientCopy = copyList(gradients); 
		Collections.reverse(gradientCopy);

		if(TorF == true) {
			weightChange = initializeList(gradientCopy); 
			initializeMomentLists(gradientCopy);
			TorF = false;
		}
	
		updateBiasedFirstMomentEstimate();
		updateBiasedSecondMomentEstimate();
		computeBiasCorrectedFirstMoment();
		computeBiasCorrectedSecondMoment();
		calculateParameterUpdate();
		return weightChange; 
	}

	private List<double[][]> copyList(List<double[][]> list) {
		List<double[][]> copy = initializeList(list);
	
		for(int i=0; i<list.size(); i++) {
			for(int j=0; j<list.get(i).length; j++) {
				for(int k=0; k<list.get(i)[0].length; k++) {
					copy.get(i)[j][k] = list.get(i)[j][k];
				}
			}
		}
		return copy;
	}

	private void initializeMomentLists(List<double[][]> gradients) {
		firstMomentEstimate = initializeList(gradients);
		secondMomentEstimate = initializeList(gradients);
		firstMomentEstimateCorrected = initializeList(gradients);
		secondMomentEstimateCorrected = initializeList(gradients);
	}
	
	private List<double[][]> initializeList(List<double[][]> list) {
		double[][] array;
		List<double[][]> newList = new ArrayList<double[][]>();

		for (int i = 0; i < list.size(); i++) {
			array = new double[list.get(i).length][list.get(i)[0].length];
			newList.add(array);
		}
		
		return newList;
	}

	
	private void updateBiasedFirstMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			for (int i = 0; i < gradientCopy.get(k).length; i++) {
				for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
					firstMomentEstimate.get(k)[i][j] = (beta1 * firstMomentEstimate.get(k)[i][j])
							+ ((1.0 - beta1) * (gradientCopy.get(k)[i][j]));
				}
			}
		}
	}

	private void updateBiasedSecondMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			for (int i = 0; i < gradientCopy.get(k).length; i++) {
				for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
					secondMomentEstimate.get(k)[i][j] = (beta2 * secondMomentEstimate.get(k)[i][j])
							+ ((1.0 - beta2) * gradientCopy.get(k)[i][j] * gradientCopy.get(k)[i][j]);
				}
			}
		}

	}

	private void computeBiasCorrectedFirstMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			for (int i = 0; i < gradientCopy.get(k).length; i++) {
				for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
					firstMomentEstimateCorrected.get(k)[i][j] = firstMomentEstimate.get(k)[i][j]
							/ (1.0 - Math.pow(beta1, betaCounter));

				}
			}
		}
	}

	private void computeBiasCorrectedSecondMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			for (int i = 0; i < gradientCopy.get(k).length; i++) {
				for (int j = 0; j < gradientCopy.get(k)[0].length; j++) {
					secondMomentEstimateCorrected.get(k)[i][j] = secondMomentEstimate.get(k)[i][j]
							/ (1.0 - Math.pow(beta2, betaCounter));

				}
			}
		}
	}


	private void calculateParameterUpdate() {
		for (int i = 0; i < gradientCopy.size(); i++) {
			for (int j = 0; j < gradientCopy.get(i).length; j++) {
				for (int k = 0; k < gradientCopy.get(i)[0].length; k++) {
					weightChange.get(i)[j][k] = ((learningRate * firstMomentEstimateCorrected.get(i)[j][k])
							/ (Math.sqrt(secondMomentEstimateCorrected.get(i)[j][k]) + offSet));
				}
			}
		}
		betaCounter++;
	}
	
}
class Basic extends Optimizer {
	
}
