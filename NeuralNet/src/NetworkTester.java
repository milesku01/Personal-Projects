
public class NetworkTester {
	static NetworkModel model = new NetworkModel(3, 1000);
	static Weights weights = new Weights(); 
	static NetworkEvaluator evaluator = new NetworkEvaluator();
	static NetworkTrainer trainer = new NetworkTrainer(); 
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		model.buildInputLayer("C:\\Users\\kuhnm\\Desktop\\Inputs.txt", 4, 2);
		model.buildHiddenLayer(10, "TANH");
		model.buildHiddenLayer(10, "TANH");
		model.buildOutputLayer(2, "SOFTMAX"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, "ADAM"); 
	
		modelSaver.saveModel(model, weights); 

	}
}
