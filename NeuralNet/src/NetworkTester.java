
public class NetworkTester {
	static NetworkModel model = new NetworkModel(100,10000);
	static Weights weights = new Weights(); 
	static NetworkEvaluator evaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	
	public static void main(String[] args) {
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 200, 1);
		model.buildHiddenLayer(20, "ELU");
		model.buildOutputLayer(1, "LINEAR");
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights); 

	}
}
