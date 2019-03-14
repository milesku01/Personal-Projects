
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		 
	
		
		//model.buildConvolutionalLayer(2, 3, 1, 1, "zero-padding", "Orange Ball Photos all");
	/*    model.buildConvolutionalLayer(6, 6, 1, 1, 3, 1, 1, "zero-padding", "ImageText2");
		model.buildReluLayer(); 
		
		model.buildPoolingLayer(2, "MAX");

		model.buildHiddenLayer("LEAKYRELU"); 
		
		model.buildOutputLayer(2, "SOFTMAX", "BallTargets"); 
	*/
		
		//model.buildInputLayer(400, 2, 400, "Inputs");
		model.buildInputLayerText(2, 16, 2, "Inputs", "Lookup");
		model.buildHiddenLayer(10, "TANH");
		model.buildOutputLayer(1, "SIGMOID");
		
		
	trainer.train(model, 100, "ADAM"); 
		//trainer.trainUntil(model, .90, 10, "MOMENTUM");
		
	modelSaver.saveModel(model, trainer.weights); // for regular train
	//	modelSaver.saveModel(model, trainer.net.weights); // for train until 

	//  modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
	
		
		
		
    	modelEvaluator.predictNCAA("basketball6", "Lookup", "GOPHERS", "GONZAGA");
		
		
	}
}
