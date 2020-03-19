import java.util.List;

public class DiagnosticTool {
	NetworkModel model = new NetworkModel();
	NetworkTrainer trainer = new NetworkTrainer(); 
	FileReader frLayers = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticLayers.txt");
	FileReader frWeights = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticWeights.txt");
	FileReader frWeightChanges = new FileReader("C:\\Users\\kuhnm\\Desktop\\NeuralNetworkDiagnosticWeightChanges.txt"); 
	
	
	public double loss = 0.6932483625355279;
	public double regularization = 1.0769364244274416E-4;
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
	//	compareWeights(trainer.weights); 
	
		
	//	compareRegularization(trainer.regularizationTerm()); 
	//	compareLoss(trainer.reportLoss(trainer.layers.get(trainer.layers.size()-1))); 
	//	compareWeightChanges(trainer.weightChanges); 
		
	//	trainer.weightList;
	//	trainer.gradients;
	//	trainer.accuracy;
		
	//	trainer.regularizationTerm();
	//	trainer.reportLoss(trainer.layers.get(trainer.layers.size()-1)); 
		
	//	trainer.weightChanges; //cast to double[][] 
		
		
	
	}

	
	private void compareLayers(List<Layer> layers) {
		int counter = 0; 
		boolean change = false; 
		List<Double> expectedValues = frLayers.readInputIntoList();
	
		System.out.println(expectedValues.size());
		
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
		}
	}
	
	private void compareWeights(Weights weights) {
		
		for(int i=0; i < weights.weightList.size(); i++) {
			System.out.println(java.util.Arrays.deepToString(weights.weightList.get(i))); 
		}
	}
	

	
	private void compareRegularization(double reg) {
		if(Math.abs(reg - regularization) > .0000001) {
			System.out.println("There is a difference in regularization"); 
		}
	}
	
	private void compareLoss(double loss) {
		if(Math.abs(this.loss - loss) > .0000001) {
			System.out.println("There is a difference in loss");
		}
	}
	
	private void compareWeightChanges(List<Object> weightChanges) {
		for (int i = 0; i < weightChanges.size(); i++) {
			if (weightChanges.get(i) instanceof double[][]) {
				System.out.println(java.util.Arrays.deepToString((double[][])weightChanges.get(i))); 
			}
		}
	}
	
	private boolean isWithinTolerance(double real, double expected) {
		if(Math.abs(this.loss - loss) > .0000001) {
			return false;
		} else {
			return true; 
		}
	}
	
}
