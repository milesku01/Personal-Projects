
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		 
		/*
		model.buildConvolutionalLayer(3, 3, 1, 1, "zero-padding", "Orange Ball Photos all");
	//	model.buildConvolutionalLayer(6, 6, 1, 1, 3, 1, 1, "zero-padding", "ImageText2");
		model.buildReluLayer(); 

		model.buildHiddenLayer("LEAKYRELU"); 
		
		model.buildOutputLayer(2, "SOFTMAX", "BallTargets"); 
	*/
		
		model.buildInputLayer(4, 2, 4, "Inputs");
		model.buildHiddenLayer(5, "TANH");
		model.buildOutputLayer(1, "LINEAR");
		
		trainer.train(model, 2, "MOMENTUM"); 
	//	trainer.trainUntil(model, .90, 8, "MOMENTUM");
		
	//	modelSaver.saveModel(model, weights);

	//  modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
    //	modelEvaluator.predictConv("model81", "ImageTest");
		
		
	}
}
