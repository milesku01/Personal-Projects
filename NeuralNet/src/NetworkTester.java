
public class NetworkTester {
	static NetworkModel model = new NetworkModel(500, 1000);
	static Weights weights = new Weights(); 
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {

		
		model.buildInputLayer("Inputs", 500, 3);
		model.buildHiddenLayer(100, "TANH");
		model.buildHiddenLayer(10, "TANH");
		model.buildOutputLayer(1, "LINEAR"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, "ADAM"); 
	
		//modelSaver.saveModel(model, weights); 
		
		
		//modelEvaluator.predict("random6", 58, 38,	98,	59,	27,	26,	18,	77,	40,	71); 
		
	//	modelEvaluator.predict("radom10", "testFile");


	}
}
