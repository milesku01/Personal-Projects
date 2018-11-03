
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
		} else if (activation.equals("SOTMAX")){
			activationObject = new Softmax();
		} else if (activation.equals("LEAKYRELU")){ 
			activationObject = new LeakyRelu(); 
		} else if (activation.equals("LINEAR")){
			activationObject = new Linear(); 
		}

	}
	
	double[][] layerValue; 
	
	public double[][] activate(Layer layer) { //used only for overriding the superclass
		createActivationObject(layer.activation); 
		layerValue = activationObject.activate(layer);
		return layerValue; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		createActivationObject(layer.activation); 
		layerValue = activationObject.computeActivatedDerivative(layer);
		return layerValue; 
	}
	
}
class Sigmoid extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = layer.layerValue; 
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = sigmoid(layerValue[i][j]);
			}
		}
		return layerValue;
	}
	private static double sigmoid(double x)
	{
	    return 1 / (1 + Math.exp(-x));
	}
	
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = sigmoid(layerValue[i][j]) * (1-layerValue[i][j]);
			}
		}
		return layerValue; 
	}
}

class Elu extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = layer.layerValue;
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
		double[][] layerValue = layer.preActivatedValue;
		
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
		double[][] layerValue = layer.layerValue;
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] = 0;
				}
			}
		}
		return layerValue; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] = 0;
				} else {
					layerValue[i][j] = 1; 
				}
			}
		return layerValue; 
	}
}
class LeakyRelu extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = layer.layerValue;
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] *= .01;
				}
			}
		}
		return layerValue; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] = .01;
				} else {
					layerValue[i][j] = 1; 
				}
			}
		return layerValue; 
	}
}

class Tanh extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = layer.layerValue;
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = Math.tanh(layerValue[i][j]);
			}
		}
		return layerValue;
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
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
		double[][] layerValue = layer.layerValue;
		return layerValue;
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = 1;
			}
		}
		return layerValue; 
	}
}
class Softmax extends Activator{
	public double[][] activate(Layer layer){
		double[][] layerValue = layer.layerValue;
		double[] sums = formatSums(layerValue); 
		
		for(int i=0; i<layerValue.length; i++) {
			for(int j=0; j<layerValue[0].length; j++) {
				layerValue[i][j] = (Math.pow(Math.E, layerValue[i][j])) / sums[i]; 
			}
		}
		return layerValue;
	}
	private double[] formatSums(double[][] layerValue){
		double[] sums = new double[layerValue.length];
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				sums[i] += Math.pow(Math.E, layerValue[i][j]);
			}
		}
		return sums; 
	}
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		return null; 
	}
}


