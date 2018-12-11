
public class NetworkTester {
	static NetworkModel model = new NetworkModel(1, 100);
	static Weights weights = new Weights(); 
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {

		
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 1, 2);
		model.buildHiddenLayer(2, "TANH");
		model.buildHiddenLayer(3, "SIGMOID");
		model.buildHiddenLayer(2, "TANH");
		model.buildHiddenLayer(3, "SIGMOID");
		model.buildOutputLayer(1, "LINEAR"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, "ADAM"); 
	
		modelSaver.saveModel(model, weights); 
		
		
		modelEvaluator.predict("model36", 1, 0); 


	}
}
