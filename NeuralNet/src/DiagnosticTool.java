import java.util.ArrayList;
import java.util.List;

public class DiagnosticTool {
	NetworkModel model = new NetworkModel();
	NetworkTrainer trainer = new NetworkTrainer(); 
	FileReader frLayers = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticLayers.txt");
	FileReader frWeights = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticWeights.txt");
	FileReader frWeightChanges = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticWeightChanges.txt"); 
	
	
	public double loss = 0.6932483625355279;
	public double regularization = 1.0769364244274416E-4;
	public boolean diagnosticSuccess = true; 
//	FileReader frWeights = new FileReader(); 
//	FileReader frGradients = new FileReader();
//	FileReader frWeightChanges = new FileReader();
	
	public void runStandardDiagnostic() {
	
		model.buildInputLayerDiagnostic(4, 2, 4, "Test");
		
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
		compareWeightChanges(trainer.weightChanges); 
		
		printDiagnosticMessage(); 
	}

	
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
	
	private void compareRegularization(double reg) {
		if(!isWithinTolerance(reg, regularization)) {
			System.out.println("There is a difference in regularization"); 
			diagnosticSuccess = false; 
		}
	}
	
	private void compareLoss(double loss) {
		if(!isWithinTolerance(this.loss, loss)) {
			System.out.println("There is a difference in loss");
			diagnosticSuccess = false; 
		}
	}
	
	private void compareWeightChanges(List<Object> weightChanges) { //TODO change from object state to double[][] after 
																//major changes
		List<double[][]> weight = new ArrayList<double[][]>();
		
		for (int i = 0; i < weightChanges.size(); i++) {
			if (weightChanges.get(i) instanceof double[][]) {
				weight.add((double[][])weightChanges.get(i)); 
			}
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
	
	private boolean isWithinTolerance(double real, double expected) {
		if(Math.abs(real - expected) > .0000001) {
			return false;
		} else {
			return true; 
		}
	}
	
	private void printDiagnosticMessage() {
		if(diagnosticSuccess) {
			System.out.println("Diagnostic was successful!"); 
		} else {
			System.out.println("There wasa discrepancy");
		}
	}
	
}
