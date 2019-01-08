
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
		NetworkTrainer trainer = new NetworkTrainer(); 
	
		model.buildInputLayer("LargeInputs", 10000, 500, 10000);
		model.buildHiddenLayer(100, "RELU");
		model.buildHiddenLayer(100, "TANH");
		model.buildOutputLayer(2, "SOFTMAX"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 11, "ADAM"); 
		
	
		//modelSaver.saveModel(model, weights); 
	
		
		//modelEvaluator.predict("fileName", 2, 2); 
		
    	//modelEvaluator.predict("fileName", "testFile");
	}
}
