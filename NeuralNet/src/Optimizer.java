import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Optimizer {
	Optimizer optimizationObject;
	List<Object> weightChange;
	public boolean TorF = true;
	public double learningRate = .000005;
	//public double learningRate = .001;
	
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

	public List<Object> optimize(List<Gradients> gradients, String optimizerString) {
		if (TorF == true) {
			createOptimizerObject(optimizerString);
			TorF = false;
		}
		weightChange = optimizationObject.optimize(gradients, "");
		return weightChange;
	}

	protected List<Object> initializeList(List<Gradients> list) {
		double[][] twoArray;
		double[][][] threeArray;
		List<double[][][]> threeArrayList;
		List<Object> newList = new ArrayList<Object>(list.size());

		for (int i = 0; i < list.size(); i++) {
			if (list.get(i).twoDGradient != null) {
				twoArray = new double[list.get(i).twoDGradient.length][list.get(i).twoDGradient[0].length];
				newList.add(twoArray);
			} else if (list.get(i).twoDGradient == null) {
				threeArrayList = new ArrayList<double[][][]>(list.get(i).threeDGradientList.size());
				
				for (int j = 0; j < list.get(i).threeDGradientList.size(); j++) {
					threeArray = new double[list.get(i).threeDGradientList.get(j).length][list.get(i).threeDGradientList
							.get(j)[0].length][list.get(i).threeDGradientList.get(j)[0][0].length];
					threeArrayList.add(threeArray);
				}
				newList.add(threeArrayList);
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

	List<Object> firstMomentEstimate;
	List<Object> secondMomentEstimate;
	List<Object> firstMomentEstimateCorrected;
	List<Object> secondMomentEstimateCorrected;
	List<Gradients> gradientCopy;

	public List<Object> optimize(List<Gradients> gradients, String optimizerString) {
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
		
	//	gradientCopy.clear();
		gradients.clear();

		return weightChange;
	}

	private void initializeMomentLists(List<Gradients> gradients) {
		firstMomentEstimate = initializeList(gradients);
		secondMomentEstimate = initializeList(gradients);
		firstMomentEstimateCorrected = initializeList(gradients);
		secondMomentEstimateCorrected = initializeList(gradients);
	}

	private void updateBiasedFirstMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {	
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) firstMomentEstimate.get(k))[i][j] = (beta1
								* ((double[][]) firstMomentEstimate.get(k))[i][j])
								+ ((1.0 - beta1) * (gradientCopy.get(k).twoDGradient[i][j]));
					}
				}
				
			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) firstMomentEstimate.get(k)).get(h)[l][i][j] = (beta1
										* ((List<double[][][]>) firstMomentEstimate.get(k)).get(h)[l][i][j])
										+ ((1.0 - beta1) * (gradientCopy.get(k).threeDGradientList.get(h)[l][i][j]));
							}
						}
					}
				}
			}
		}
	}

	private void updateBiasedSecondMomentEstimate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) secondMomentEstimate.get(k))[i][j] = (beta2
								* ((double[][]) secondMomentEstimate.get(k))[i][j])
								+ ((1.0 - beta2) * gradientCopy.get(k).twoDGradient[i][j]
										* gradientCopy.get(k).twoDGradient[i][j]);
					}
				}
			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) secondMomentEstimate.get(k)).get(h)[l][i][j] = (beta2
										* ((List<double[][][]>) secondMomentEstimate.get(k)).get(h)[l][i][j])
										+ ((1.0 - beta2) * gradientCopy.get(k).threeDGradientList.get(h)[l][i][j]
												* gradientCopy.get(k).threeDGradientList.get(h)[l][i][j]);
							}
						}
					}
				}
			}
		}

	}

	private void computeBiasCorrectedFirstMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) firstMomentEstimateCorrected
								.get(k))[i][j] = ((double[][]) firstMomentEstimate.get(k))[i][j]
										/ (1.0 - Math.pow(beta1, betaCounter));
					}
				}

			} 
			else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) firstMomentEstimateCorrected.get(k)).get(
										h)[l][i][j] = ((List<double[][][]>) firstMomentEstimate.get(k)).get(h)[l][i][j]
												/ (1.0 - Math.pow(beta1, betaCounter));
							}
						}
					}
				}
			}
		}
	}

	private void computeBiasCorrectedSecondMoment() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) secondMomentEstimateCorrected
								.get(k))[i][j] = ((double[][]) secondMomentEstimate.get(k))[i][j]
										/ (1.0 - Math.pow(beta2, betaCounter));
					}
				}

			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) secondMomentEstimateCorrected.get(k)).get(
										h)[l][i][j] = ((List<double[][][]>) secondMomentEstimate.get(k)).get(h)[l][i][j]
												/ (1.0 - Math.pow(beta2, betaCounter));
							}
						}
					}
				}
			}
		}
	}

	private void calculateParameterUpdate() {
		for (int k = 0; k < gradientCopy.size(); k++) {
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) weightChange.get(k))[i][j] = ((learningRate
								* ((double[][]) firstMomentEstimateCorrected.get(k))[i][j])
								/ (Math.sqrt(((double[][]) secondMomentEstimateCorrected.get(k))[i][j]) + offSet));
					}
				}
			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) weightChange.get(k)).get(h)[l][i][j] = ((learningRate
										* ((List<double[][][]>) firstMomentEstimateCorrected.get(k)).get(h)[l][i][j])
										/ (Math.sqrt(((List<double[][][]>) secondMomentEstimateCorrected.get(k))
												.get(h)[l][i][j]) + offSet));
							}
						}
					}
				}
				
				//System.out.println(java.util.Arrays.deepToString(((List<double[][][]>)weightChange.get(0)).get(0)));
			}
		}
		

		betaCounter++;
	}

}

class Basic extends Optimizer {

	List<Gradients> gradientCopy;

	public List<Object> optimize(List<Gradients> gradients, String optimizerString) {
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
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) weightChange.get(k))[i][j] = learningRate
								* (gradientCopy.get(k).twoDGradient[i][j]);
					}
				}

			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) weightChange.get(k)).get(h)[l][i][j] = learningRate
										* (gradientCopy.get(k).threeDGradientList.get(h)[l][i][j]);
							}
						}
					}
				}
			}
		}
	}

}

class Momentum extends Optimizer {
	
	double beta = .9; 
	List<Gradients> gradientCopy;

	public List<Object> optimize(List<Gradients> gradients, String optimizerString) {
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
			if (gradientCopy.get(k).twoDGradient != null) {
				for (int i = 0; i < gradientCopy.get(k).twoDGradient.length; i++) {
					for (int j = 0; j < gradientCopy.get(k).twoDGradient[0].length; j++) {
						((double[][]) weightChange.get(k))[i][j] = beta*((double[][])weightChange.get(k))[i][j] + (learningRate) * (gradientCopy.get(k).twoDGradient[i][j]);
					}
				}

			} else if (gradientCopy.get(k).twoDGradient == null) {
				for (int h = 0; h < gradientCopy.get(k).threeDGradientList.size(); h++) {
					for (int l = 0; l < gradientCopy.get(k).threeDGradientList.get(h).length; l++) {
						for (int i = 0; i < gradientCopy.get(k).threeDGradientList.get(h)[0].length; i++) {
							for (int j = 0; j < gradientCopy.get(k).threeDGradientList.get(h)[0][0].length; j++) {
								((List<double[][][]>) weightChange.get(k)).get(h)[l][i][j] = beta*((List<double[][][]>) weightChange.get(k)).get(h)[l][i][j] + learningRate*(gradientCopy.get(k).threeDGradientList.get(h)[l][i][j]);
							}
						}
					}
				}
			}
		}
	}
	
}


