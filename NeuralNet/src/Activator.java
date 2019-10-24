
public class Activator {
	Activator activationObject; 
	
	public void createActivationObject(String activation) {
		if(activation.equals("RELU")) {
			activationObject = new Relu();
		} else if (activation.equals("ELU")){
			activationObject = new Elu(); 
		} else if (activation.equals("SIGMOID")){
			activationObject = new Sigmoid();
		} else if (activation.equals("TANH")){
			activationObject = new Tanh();
		} else if (activation.equals("SOFTMAX")){
			activationObject = new Softmax();
		} else if (activation.equals("LEAKYRELU")){ 
			activationObject = new LeakyRelu(); 
		} else if (activation.equals("LINEAR")){
			activationObject = new Linear(); 
		}

	}
	
	public int convertActivationString(String activation) {
		int integerIdentifier = 0; 
		if(activation.equals("RELU")) {
			integerIdentifier = 1; 
		} else if (activation.equals("ELU")){
			integerIdentifier = 2; 
		} else if (activation.equals("SIGMOID")){
			integerIdentifier = 3; 
		} else if (activation.equals("TANH")){
			integerIdentifier = 4; 
		} else if (activation.equals("SOFTMAX")){
			integerIdentifier = 5; 
		} else if (activation.equals("LEAKYRELU")){ 
			integerIdentifier = 6; 
		} else if (activation.equals("LINEAR")){
			integerIdentifier = 7; 
		}
		return integerIdentifier; 
	}
	
	public String convertActivationInt(int actInt) {
		String identifier = ""; 
		if(actInt == 1) {
			identifier = "RELU"; 
		} else if (actInt == 2){
			identifier = "ELU"; 
		} else if (actInt == 3){
			identifier = "SIGMOID"; 
		} else if (actInt == 4){
			identifier = "TANH";  
		} else if (actInt == 5){
			identifier = "SOFTMAX";  
		} else if (actInt == 6){ 
			identifier = "LEAKYRELU";  
		} else if (actInt == 7){
			identifier = "LINEAR"; 
		}
		return identifier;
	}
	
	double[][] layerValue; 
	
	public double[][] activate(Layer layer) { //used only for overriding the superclass
		createActivationObject(layer.activation); 
		return activationObject.activate(layer);
	}
	public double[][] computeActivatedDerivative(Layer layer){
		createActivationObject(layer.activation); 
		layerValue = activationObject.computeActivatedDerivative(layer);
		return layerValue; 
	}
	public double[][] copyArray(double[][] input) {
		double[][] copy = new double[input.length][input[0].length]; 
		for(int i=0; i<input.length; i++) {
			for(int j=0; j<input[0].length; j++) {
				copy[i][j] = input[i][j];
			}
		}
		return copy;
	}
}

class Sigmoid extends Activator{
	public double[][] activate(Layer layer){
		//double[][] layerValue = copyArray(layer.layerValue);
		for(int i = 0; i < layer.layerValue.length; i++) {
			for (int j = 0; j < layer.layerValue[0].length; j++) {
				layer.layerValue[i][j] = sigmoid(layer.layerValue[i][j]);
			}
		}
		return layer.layerValue;
	}
	private static double sigmoid(double x)
	{
	    return (1.0 / (double)(1.0 + Math.exp(-x)));
	}
	
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = copyArray(layer.preActivatedValue);
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = sigmoid(layerValue[i][j]) * (1.0-sigmoid(layerValue[i][j]));
			}
		}
		return layerValue; 
	}
}

class Elu extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = copyArray(layer.layerValue);
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j]  = (Math.pow(Math.E, layerValue[i][j])) + 1;
				}
			}
		}
		return layerValue; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = copyArray(layer.preActivatedValue);
		
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] = Math.pow(Math.E, (layerValue[i][j]));
				} else { //may get weird results of layerValue[i][j] == 0
					layerValue[i][j] = 1; 
				}
			}
		}
		return layerValue; 
	}
}
class Relu extends Activator{
	public double[][] activate(Layer layer){
	//	double[][] layerValue = copyArray(layer.layerValue);
		for(int i=0; i < layer.layerValue.length; i++) {
			for(int j=0; j < layer.layerValue[0].length; j++) {
				if(layer.layerValue[i][j] <= 0) {
					layer.layerValue[i][j] = 0;
				}
			}
		}
		return layer.layerValue; 
	}
	
	public double[][] computeActivatedDerivative(Layer layer){
	//	double[][] layerValue = copyArray(layer.preActivatedValue);
		for(int i=0; i < layer.preActivatedValue.length; i++) {
			for(int j=0; j < layer.preActivatedValue[0].length; j++) {
				if(layer.preActivatedValue[i][j] <= 0) {
					layer.preActivatedValue[i][j] = 0;
				} else {
					layer.preActivatedValue[i][j] = 1; 
				}
			}
		}
		return layer.preActivatedValue; 
	}
}
class LeakyRelu extends Activator{
	public double[][] activate(Layer layer){
		//double[][] layerValue = copyArray(layer.layerValue);
		for(int i=0; i < layer.layerValue.length; i++) {
			for(int j=0; j < layer.layerValue[0].length; j++) {
				if(layer.layerValue[i][j] < 0) {
					layer.layerValue[i][j] *= .01;
				}
			}
		}
		return layer.layerValue; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		//double[][] layerValue = copyArray(layer.preActivatedValue);
		for(int i=0; i < layer.preActivatedValue.length; i++) {
			for(int j=0; j < layer.preActivatedValue[0].length; j++) {
				if(layer.preActivatedValue[i][j] < 0) {
					layer.preActivatedValue[i][j] = .01;
				} else {
					layer.preActivatedValue[i][j] = 1; 
				}
			}
		}
		return layer.preActivatedValue; 
	}
}

class Tanh extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = copyArray(layer.layerValue);
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = Math.tanh(layerValue[i][j]);
			}
		}
		return layerValue;
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = copyArray(layer.preActivatedValue);
		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = (1 - (Math.pow(Math.tanh(layerValue[i][j]), 2)));
			}
		}
		return layerValue; 
	}
}

class Linear extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = copyArray(layer.layerValue);
		return layerValue;
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = copyArray(layer.preActivatedValue);
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = 1;
			}
		}
		return layerValue; 
	}
}
class Softmax extends Activator{
	double[] max; 

	public double[][] activate(Layer layer) {
		double[][] layerValue = copyArray(layer.layerValue);
		max = new double[layerValue.length];
		max = getMax(layerValue); 
		double[] sums = formatSums(layerValue); 
		
	
		for(int i=0; i<layerValue.length; i++) {
			for(int j=0; j<layerValue[0].length; j++) {
				layerValue[i][j] = (Math.pow(Math.E, (layerValue[i][j]-max[i]))) / sums[i]; 
			}
		}
		return layerValue;
	}
	private double[] formatSums(double[][] layerValue){
		double[] sums = new double[layerValue.length];
		
		for(int i=0; i < layerValue.length; i++) { 
			for(int j=0; j < layerValue[0].length; j++) {
				sums[i] += Math.pow(Math.E, (layerValue[i][j]-max[i]));
			}

		}
		return sums; 
	}
	
	private double[] getMax(double[][] layerValue) {
		double constant = 0; 
		for(int i=0; i<layerValue.length; i++) {
			constant = layerValue[i][0];
			for(int j=1; j<layerValue[0].length; j++) {
				if(layerValue[i][j] > constant) {
					constant = layerValue[i][j];
				}
			}
			max[i] = constant; 
		}
		return max; 
	}
	
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = copyArray(layer.layerValue); //doesn't require previous layerValue **special case 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				layerValue[i][j] = (layerValue[i][j])*(1.0-layerValue[i][j]);
			}
		}
		return layerValue; 
	}
}


