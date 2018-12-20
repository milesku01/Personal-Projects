
public class NetworkTester {
	static NetworkModel model = new NetworkModel(4, 5);
	static Weights weights = new Weights(); 
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {

		
		model.buildInputLayer("Inputs", 4, 2);
		model.buildHiddenLayer(2, "RELU");
		model.buildOutputLayer(1, "LINEAR"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, "ADAM"); 
	
		modelSaver.saveModel(model, weights); 
		
		
		
		
	//	modelEvaluator.predict("random17", 199600	,-1, 	12); 
		
		//modelEvaluator.predict("random14", "testFile");


	}
}
