
public class NetworkTester {
	static NetworkModel model = new NetworkModel(500, 400);
	static Weights weights = new Weights(); 
	static NetworkEvaluator evaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	
	public static void main(String[] args) {
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 500, 3);
		model.buildHiddenLayer(3, "TANH");
		model.buildHiddenLayer(2, "TANH");
		model.buildOutputLayer(1, "TANH");
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights); 

	}
}
