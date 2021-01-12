import java.util.ArrayList;
import java.util.List;
/**
 * Class DiagnosticTool is used to find any changes in output of a previous known working network
 * and a run through of the current network
 *
 */
public class DiagnosticTool {
	
	NetworkModel model = new NetworkModel();
	NetworkTrainer trainer = new NetworkTrainer(); 
	FileReader frLayers = new FileReader(System.getProperty("user.home") + "\\Desktop\\Diagnostic\\NeuralNetworkDiagnosticLayers.txt");
	FileReader frWeights = new FileReader(System.getProperty("user.home") + "\\Desktop\\Diagnostic\\NeuralNetworkDiagnosticWeights.txt");
	FileReader frWeightChanges = new FileReader(System.getProperty("user.home") + "\\Desktop\\Diagnostic\\NeuralNetworkDiagnosticWeightChanges.txt"); 
	
	public double loss = 0.6932542744391355;
	public double regularization = 1.0827903295037047E-4;
	public boolean diagnosticSuccess = true; 

	/**
	 * void runStandardDiagnostic runs a simple diagnostic test run from a Diagnostic object
	 * creates a model with may layers and all the types of activation functions
	 * The only difference from a standard model and this one is diagnostic input layer which is needed for proper 
	 * weight setup 
	 * 
	 * The values of the layers, the weights, the weight changes, the loss and regularization are compared to a previous model
	 * 
	 * Based on those comparisons, an appropriate diagnostic message is displayed 
	 */
	public void runStandardDiagnostic() {
	
		model.buildInputLayerDiagnostic(4, 2, 4, "DiagnosticTest");
		
		model.buildHiddenLayer(3, "RELU");
			
		model.buildHiddenLayer(3, "ELU");
			
		model.buildHiddenLayer(3, "SIGMOID"); 
			
		model.buildHiddenLayer(3, "TANH");
		
		model.buildHiddenLayer(3, "LEAKYRELU");
			
		model.buildHiddenLayer(3, "LINEAR"); 
		
		model.buildOutputLayer(2, "SOFTMAX");
			
		trainer.train(model, 10, "ADAM"); 
		
		compareLayers(trainer.layers); 
		compareWeights(trainer.weights.weightList); 
		
		compareRegularization(trainer.regularizationTerm()); 
		compareLoss(trainer.reportLoss(trainer.layers.get(trainer.layers.size()-1))); 
		compareWeightChanges(BackPropagator.weightChanges); 
		
		printDiagnosticMessage(); 
	}

	/**
	 * void compareLayers compares the layerValues of all the layers in the model after 10 epochs
	 * The expected values are read from a file into a list 
	 * 
	 * Nested for loops are used to individually compare the value of the layers and the expected values in the list, this requires the use
	 * of the counter. In each iteration of the for loop a change is checked for using the isWithinTolerance which returns true if the values match
	 * If the values don't match then boolean change is changed to true 
	 * 
	 * If change is true then boolean diagnostic success becomes false and the message that a layerValue has changed is printed
	 * 
	 * @param layers: The list of layers tracked by the network trainer is passed to this method 
	 */
	private void compareLayers(List<Layer> layers) {
		int counter = 0; 
		boolean change = false; 
		List<Double> expectedValues = frLayers.readInputIntoList();
	
		for(int i=0; i < layers.size(); i++) {
			for(int j=0; j < layers.get(i).layerValue.length; j++) {
				for(int k=0; k < layers.get(i).layerValue[0].length; k++) {
					if(!isWithinTolerance(layers.get(i).layerValue[j][k], expectedValues.get(counter))) {
						 change = true; 	 
					}
					counter++; 
				}
			}
		}
		if(change) {
			System.out.println("A layerValue has changed"); 
			diagnosticSuccess = false; 
		}
	}
	
	/**
	 * void compareWeights compares the weight values of all the layers in the model after 10 epochs 
	 * The expectedValues are read from a file into a list 
	 * 
	 * Nested for loops are used to individually compare the weight values and the expected values from the list
	 * This is done using a counter to get the values from the list. The expected values are compared with the weight values 
	 * tracked by the weightList in the trainer object using the isWithinTolerance method 
	 * 
	 * If change is true (a value was not within tolerance) then the diagnosticSuccess boolean is tripped to
	 * false indicating an unsuccessful diagnostic 
	 * 
	 * @param weights: The list of weight values tracked by the NetworkTrainer object is passed to this method 
	 */
	private void compareWeights(List<double[][]> weights) {
		int counter = 0; 
		boolean change = false; 
		List<Double> expectedValues = frWeights.readInputIntoList();
	
		for(int i=0; i < weights.size(); i++) {
			for(int j=0; j < weights.get(i).length; j++) {
				for(int k=0; k < weights.get(i)[0].length; k++) {
					if(!isWithinTolerance(weights.get(i)[j][k], expectedValues.get(counter))) {
						 change = true; 
					 
					}
					counter++; 
				}
			}
		}
		if(change) {
			System.out.println("A weight has changed"); 
			diagnosticSuccess = false; 
		}
	}
	
	/**
	 * void compareRegularization checks if there is a difference between the known regularization and the regularizaton produced by the
	 * network in its current config using the isWithinTolerance method
	 * @param reg: the regularizaiton produced by the model is passed to this method for comparisong
	 */
	private void compareRegularization(double reg) {
		if(!isWithinTolerance(reg, regularization)) {
			System.out.println("There is a difference in regularization"); 
			diagnosticSuccess = false; 
		}
	}
	
	/**
	 * void compareLoss checks if there is a difference between the known loss of a previous model and the loss produced
	 * by the current model 
	 * @param loss: the loss produced by the model is passed to this method for comparison
	 */
	private void compareLoss(double loss) {
		if(!isWithinTolerance(this.loss, loss)) {
			System.out.println("There is a difference in loss " + loss);
			diagnosticSuccess = false; 
		}
	}
	
	//TODO document once weight changes type changes from Object to double[][]
	private void compareWeightChanges(List<double[][]> weightChanges) { //TODO change from object state to double[][] after 														//major changes
		List<double[][]> weight = new ArrayList<double[][]>();
		
		for (int i = 0; i < weightChanges.size(); i++) {
			
			weight.add(weightChanges.get(i)); 
			
		}
		
		int counter = 0; 
		boolean change = false; 
		List<Double> expectedValues = frWeightChanges.readInputIntoList();
	
		for(int i=0; i < weight.size(); i++) {
			for(int j=0; j < weight.get(i).length; j++) {
				for(int k=0; k < weight.get(i)[0].length; k++) {
					if(!isWithinTolerance(weight.get(i)[j][k], expectedValues.get(counter))) {
						 change = true;  
					}
					counter++; 
				}
			}
		}
		if(change) {
			System.out.println("A weightChange has changed"); 
			diagnosticSuccess = false; 
		}	
	}
	
	/**
	 * boolean isWithinTolerance checks if a value is within a certain distance from another, if it is, the method 
	 * returns true, if not, it returns false 
	 * @param real: the real value newly produced by the model
	 * @param expected: the expected value from the file
	 * @return: returns true if the values were within tolerance and false if they weren't
	 */
	private boolean isWithinTolerance(double real, double expected) {
		if(Math.abs(real - expected) > .0000001) {
			return false;
		} else {
			return true; 
		}
	}
	
	/**
	 * prints a successful diagnostic message is the diagnosticSuccess boolean wasn't flagged and
	 * message about a discrepancy if the diagnosticSuccess boolean was flagged
	 */
	private void printDiagnosticMessage() {
		if(diagnosticSuccess) {
			System.out.println("Diagnostic was successful!"); 
		} else {
			System.out.println("There was a discrepancy");
		}
	}
	
}
