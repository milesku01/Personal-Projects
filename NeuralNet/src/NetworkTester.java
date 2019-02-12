
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
		NetworkTrainer trainer = new NetworkTrainer(); 
	
		
		
		model.buildConvolutionalLayer(16, 3, 1, 1, "zero-padding", "Orange Ball Photos all");
	//	model.buildConvolutionalLayer(6, 6, 1, 1, 3, 1, 1, "zero-padding", "ImageText");
		model.buildReluLayer(); 
	//	model.buildPoolingLayer(2, "MAX"); 
		model.buildPoolingLayer(2, "MAX");
		
	//	model.buildPoolingLayer(2, "MAX"); 
		
		model.buildHiddenLayer("TANH"); 
	//	model.buildHiddenLayer(1, "TANH");
	//	model.buildHiddenLayer(10, "TANH");
		
		model.buildOutputLayer(2, "SOFTMAX", "BallTargets"); 
	
		
		
//		model.buildInputLayer(2, 2, 1, "Inputs");
//		model.buildHiddenLayer(5, "TANH"); 
//		model.buildOutputLayer(2, "SOFTMAX");
		

		weights.generateInitialWeights(model);
		trainer.train(model, weights, 10, "ADAM"); 
		
		modelSaver.saveModel(model, weights);
	
	
	
		//modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
    	modelEvaluator.predictConv("avg3", "testFile");
		
		
	}
}

