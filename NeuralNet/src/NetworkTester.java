
public class NetworkTester {
	static NetworkModel model = new NetworkModel(50, 1000);
	static Weights weights = new Weights(); 
	static NetworkEvaluator evaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	
	public static void main(String[] args) {
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 50, 1);
		model.buildHiddenLayer(2, "ELU"); 
		model.buildOutputLayer(1, "TANH");
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights); 

	}
}
