
public class NetworkTester {
	static NetworkModel model = new NetworkModel();
	static Weights weights = new Weights(); 
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {

		model.buildInputLayer("Inputs", 4, 2, 2);
		model.buildHiddenLayer(100, "RELU");
		model.buildHiddenLayer(100, "RELU");
		model.buildOutputLayer(2, "SOFTMAX"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 100, "ADAM"); 
	
	//	modelSaver.saveModel(model, weights); 
		
		
		
		
	//	modelEvaluator.predict("random17", 199600	,-1, 	12); 
		
		//modelEvaluator.predict("random14", "testFile");


	}
}
