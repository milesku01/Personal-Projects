
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
	
		model.buildInputLayer("Inputs", 250, 200, 250);
		model.buildHiddenLayer(50, "TANH");
		model.buildHiddenLayer(50, "TANH");
		model.buildOutputLayer(2, "SOFTMAX"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 100, "ADAM"); 
	
		//modelSaver.saveModel(model, weights); 
		
		//modelEvaluator.predict("fileName", 	2, 2); 
		
    	//modelEvaluator.predict("fileName", "testFile");
	}
}
