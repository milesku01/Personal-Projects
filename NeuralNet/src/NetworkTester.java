
public class NetworkTester {
	static NetworkModel model = new NetworkModel(3, 2); 
	static Weights weights = new Weights(); 
	static NetworkEvaluator evaluator = new NetworkEvaluator(); //used for single tests
	static NetworkTrainer trainer = new NetworkTrainer(); 
	
	public static void main(String[] args) {
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 4, 2);
		model.buildHiddenLayer(2, "RELU");
		model.buildOutputLayer(1, "SIGMOID");
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights); 

	}
}
