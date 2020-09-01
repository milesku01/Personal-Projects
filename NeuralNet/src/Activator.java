/**
 * Class Activator computes the activations of the two-dimensional arrays to potentially 
 * transform the function of the neural net to non-linear
 *
 * @Miles Kuhn
 */

public class Activator {
	
	/**
	 * the activation object is customized based on the activation string
	 * thus the activation works polymorphically
	 */
	Activator activationObject; 
	
	/**
	 * creates the appropriate object based on the activation string 
	 * so that child classes may be used in activation
	 * 
	 * @param activation string that each layer contains (assigned in constructor)
	 */
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
	
	
	/**
	 * converts the activation string into an integer used in model saving
	 * 
	 * @param activation: activation string that each layer contains
	 * @return returns the integer that represents the activation type
	 */
	public static int convertActivationString(String activation) {
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
	
	
	/**
	 * converts the activation integer into an identifier string which is used model reading 
	 * and evaluation
	 * 
	 * @param actInt: integer read from the modelFile 
	 * @return returns the activation string so it may used to reconstruct the saved model using 
	 * existing code
	 */
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
	
	/**
	 * returns the activated two-dimensional array which is assigned to the nextLayers' layerValue
	 * used in forwardPropagation
	 * 
	 * @param layer: Layer object that needs to be activated 
	 * @return returns the two-dimensional array activated on
	 */
	public double[][] activate(Layer layer) { //used only for overriding the superclass
		createActivationObject(layer.activation); 
		return activationObject.activate(layer);
	}
	
	/**
	 * returns the derivative of the preActivated layerValue for backPropagation 
	 * because we need to keep a copy of the preActivated values for backPropagation we 
	 * make copies of the preActivated values because that is what is operated on in both cases
	 * and then aliasing is no longer an issue
	 * 
	 * 
	 * @param layer: Layer object used in backPropagation
	 * @return returns the derivative of the preActivated layerValue
	 */
	public double[][] computeActivatedDerivative(Layer layer){
		createActivationObject(layer.activation); 
		double[][] layerValue = activationObject.computeActivatedDerivative(layer);
		return layerValue; 
	}

}

/**
 * Class Sigmoid is a child class of Activator and a type of activation
 * that normalizes the data between zero and one. Sigmoid is a logistic function
 * It is not used very often in practice because of the vanishing gradient problem
 *
 */
class Sigmoid extends Activator{
	
	/**
	 * Overloaded function activate that applies the activation function to each element in
	 * the two dimensional array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * each element in the preActivated array is then activated using the sigmoid function using
	 * a nested for loop
	 * 
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = sigmoid(layerValue[i][j]);
			}
		}
		return layerValue;
	}
	
	/**
	 * Applies the sigmoid function to an element of the layerValue
	 * 
	 * @param passes the double value from the preActivated array to be activated
	 * @return returns the value of the double after applying the sigmoid activation
	 */
	private double sigmoid(double x)
	{
	    return (1.0 / (1.0 + Math.exp(-x)));
	}
	
	/**
	 * Overloaded function that applies the derivative of the sigmoid function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivated value of the previous layer
	 * because this array may now be freely modified
	 * 
	 * the derivative each element in the preActivated array is calculated in a nested loop
	 * 
	 */
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = sigmoid(layerValue[i][j]) * (1.0 - sigmoid(layerValue[i][j]));
			}
		}
		return layerValue; 
	}
}

/**
 * Class Elu is a child class of Activator which activates the data as itself above 
 * 0, and is a slowly decaying exponential function below 0 to combat the dying neuron 
 * problem
 *
 */
class Elu extends Activator{
	
	/**
	 * Overloaded function that activates each element of the preActivated value array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * each element in the preActivated array is then activated using the elu function with
	 * a nested for loop
	 * 
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j]  = (Math.pow(Math.E, layerValue[i][j])) + 1;
				}
			}
		}
		return layerValue; 
	}
	
	/**
	 * Overloaded function that applies the derivative of the elu function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivatedValue which may now be freely 
	 * modified
	 * 
	 * the derivative each element in the preActivated array is calculated in a nested loop using 
	 * a conditional to adjust the derivative in a piecewise way which is dependent on whether 
	 * or not the value at that location is less than 0 
	 */
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] = Math.pow(Math.E, (layerValue[i][j]));
				} else { 
					layerValue[i][j] = 1.0; 
				}
			}
		}
		return layerValue; 
	}
}

/**
 * class Relu is a child class of parent Activator and activates the layerValue 
 * by setting any negative value in the array to 0. This activation is very quick and is the 
 * main activation used however Relu has dead neuron problem
 *
 */
class Relu extends Activator{
	
	/**
	 * Overloaded function that activates each element of the preActivated value array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * each element in the preActivated array is then activated using the Relu function with
	 * a nested for loop
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] <= 0) {
					layerValue[i][j] = 0;
				}
			}
		}
		return layerValue; 
	}
	
	/**
	 * Overloaded function that applies the derivative of the Relu function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivatedValue which may now be freely 
	 * modified
	 * 
	 * the derivative each element in the preActivated array is calculated in a nested loop using 
	 * a conditional to adjust the derivative in a piecewise way which is dependent on whether 
	 * or not the value at that location is less than 0 
	 */
	public double[][] computeActivatedDerivative(Layer layer){
		double[][] layerValue = layer.preActivatedValue;
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] <= 0) {
					layerValue[i][j] = 0;
				} else {
					layerValue[i][j] = 1; 
				}
			}
		}
		return layerValue; 
	}
}

/**
 * class LeakyRelu is a child class of parent Activator that activates the values
 * by multiplying values of layerValue less than 0 by a small factor (.01) to combat the dying 
 * neuron problem of Relu 
 *
 */
class LeakyRelu extends Activator{
	
	/**
	 * Overloaded function that activates each element of the preActivated value array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * each element in the preActivated array is then activated using the LeakyRelu function with
	 * a nested for loop and a simple conditional that multiplies layer value by the factor .01
	 * whenever the initial value is negative 
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				if(layerValue[i][j] < 0) {
					layerValue[i][j] *= .01;
				}
			}
		}
		return layerValue; 
	}
	
	/**
	 * Overloaded function that applies the derivative of the LeakyRelu function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivatedValue which may now be freely 
	 * modified
	 * 
	 * the derivative each element in the preActivated array is calculated in a nested loop using 
	 * a conditional to adjust the derivative in a piecewise way which is dependent on whether 
	 * or not the value at that location is less than 0 
	 */
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
		}
		return layerValue; 
	}
}

/**
 * 
 * Class Tanh is a child class of parent Activator that activates the using 
 * the hyperbolic tangent function which forces the values of layerValue to -1 and 1 
 *
 */
class Tanh extends Activator{
	
	/**
	 * Overloaded function activate that applies the activation function to each element in
	 * the two dimensional array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * each element in the preActivated array is then activated using the hyperbolic tangent function using
	 * a nested for loop
	 * 
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		for(int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				layerValue[i][j] = Math.tanh(layerValue[i][j]);
			}
		}
		return layerValue;
	}
	
	/**
	 * Overloaded function that applies the derivative of the tanh function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivated value of the previous layer
	 * because this array may now be freely modified
	 * 
	 * the derivative each element in the preActivated array is calculated in a nested loop
	 */
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

/**
 * Class Linear is a child of parent class Activator and "activates"
 * layer value in a linear way, i.e. the values are unchanged 
 */
class Linear extends Activator{
	
	/**
	 * Overloaded function activate that applies the activation function to each element in
	 * the two dimensional array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * layer is returned as is because the values do not changed 
	 */
	public double[][] activate(Layer layer){
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
		return layerValue;
	}

	/**
	 * Overloaded function that applies the derivative of the linear function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivated value of the previous layer
	 * because this array may now be freely modified
	 * 
	 * each element in the array is assigned to the derivative of the linear function (1.0) 
	 */
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

/**
 * Class Softmax is a child class of parent Activator
 * This activator is only to be used in the final layer (output)
 * of the network
 * Softmax normalizes the outputs of the network into probabilities
 * 
 */
class Softmax extends Activator{
	double[] max; 

	/**
	 * Overloaded function activate that applies the activation function to each element in
	 * the two dimensional array
	 * 
	 * layerValue is assigned to the reference of a copy of the preActivated value of the previous layer
	 * so that aliasing does not become a problem
	 * 
	 * first the max value form the array must be determined using the getMax function
	 * the max is then subtracted from each element so there is no out of bounds error during
	 * exponentiation
	 * 
	 * then the sums of e raised to the power layerValue at index is calculated to represent the total possible outcomes
	 * 
	 * each element of layerValue is then given a probability by dividing e to the power index by 
	 * the sum to represent the probability
	 */
	public double[][] activate(Layer layer) {
		double[][] layerValue = Utility.copyArray(layer.preActivatedValue);
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
	
	/**
	 * calculates the summation of the e raised to the power of all the elements of layerValue
	 * to represent the total possible outcomes
	 * 
	 * @return : returns the sum of e raised to the elements of layerValue (normalized with the 
	 * max value)
	 */
	private double[] formatSums(double[][] layerValue){
		double[] sums = new double[layerValue.length];
		
		for(int i=0; i < layerValue.length; i++) { 
			for(int j=0; j < layerValue[0].length; j++) {
				sums[i] += Math.pow(Math.E, (layerValue[i][j]-max[i]));
			}
		}
		return sums; 
	}
	
	/**
	 * checks the value of each column and compares it to the previous value in the column and
	 * then adds the largest one to the array "max" 
	 * 
	 * @return : returns the maximum value of each of the columns of layerValue so that they 
	 * can normalize the output probabilities
	 */
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
	
	 /**
	 * Overloaded function that applies the derivative of the Softmax function to 
	 * each element of the layers' preActivated values so that the derivatives may
	 * be multiplied together
	 * 
	 * layerValue is assigned to the reference of the preActivated value of the previous layer
	 * because this array may now be freely modified 
	 */
	public double[][] computeActivatedDerivative(Layer layer){ //TODO: check if copy of layerValue is needed 
		double[][] layerValue = Utility.copyArray(layer.layerValue); //doesn't require previous layerValue **special case 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length; j++) {
				layerValue[i][j] = (layerValue[i][j])*(1.0-layerValue[i][j]);
			}
		}
		return layerValue; 
	}
}


