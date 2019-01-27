
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
		NetworkTrainer trainer = new NetworkTrainer(); 
	
		/*
		model.buildConvolutionalLayer(3, 3, 1, "zero-padding", "Images");
		model.buildReluLayer(); 
		model.buildPoolingLayer(2, "MAX"); //String does nothing for now
		
		model.buildHiddenLayer("RELU"); 
		model.buildHiddenLayer(10, "RELU");
		model.buildOutputLayer(4, "LINEAR", "Targets"); 
		*/
		
		model.buildInputLayer(4, 2, 3, "Inputs");
		model.buildHiddenLayer(10, "TANH");
		model.buildHiddenLayer(10, "TANH");
		model.buildOutputLayer(1, "LINEAR");
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 1000, "ADAM"); 
		
		//modelSaver.saveModel(model, weights);
		
		//modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
    	//modelEvaluator.predict("save2", "testFile");
	}
}
