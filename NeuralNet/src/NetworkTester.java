
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
		NetworkTrainer trainer = new NetworkTrainer(); 
	
		
		//model.buildConvolutionalLayer(3, 3, 1, 1, "zero-padding", "Images");
		model.buildConvolutionalLayer(28, 28, 1, 2, 3, 1, 1, "zero-padding", "ImageText");
		model.buildReluLayer(); 
		//model.buildPoolingLayer(2, "MAX"); //String does nothing for now
	//	model.buildPoolingLayer(2, "MAX");
	//	model.buildPoolingLayer(2, "MAX");
		
		model.buildHiddenConvolutionalLayer(3, 3, 1, "zero-padding");
		model.buildReluLayer();
		
		model.buildHiddenLayer("TANH"); 
		//model.buildHiddenLayer(100, "TANH");
		model.buildOutputLayer(10, "SOFTMAX", "Targets"); 
		
		
		/*
		model.buildInputLayer(2, 2, 1, "Inputs");
		model.buildHiddenLayer(5, "TANH");
		model.buildOutputLayer(1, "LINEAR");
		*/
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 1000, "ADAM"); 
		
		//modelSaver.saveModel(model, weights);
		
		//modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
    	//modelEvaluator.predict("save2", "testFile");
	}
}

