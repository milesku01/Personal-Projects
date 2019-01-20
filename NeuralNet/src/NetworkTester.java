
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		Weights weights = new Weights(); 
		NetworkTrainer trainer = new NetworkTrainer(); 
	
		model.buildConvolutionalLayer(1, 3, 1, "zero-padding", "Images");
		model.buildReluLayer(); 
		model.buildPoolingLayer(2, "MAX"); //String does nothing for now
		
		model.buildHiddenConvolutionalLayer(1, 3, 1, "zero-padding");
		model.buildReluLayer(); 
		model.buildPoolingLayer(2, "MAX");
		
		model.buildHiddenLayer("RELU");  //numofInput problem
		model.buildOutputLayer(1, "LINEAR", "Targets"); 
		
		weights.generateInitialWeights(model);
		trainer.train(model, weights, 5, "ADAM"); 
		
		//modelSaver.saveModel(model, weights);
		
		//modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
    	//modelEvaluator.predict("save2", "testFile");
	}
}
