
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		 
		String name = ""; 
		
		//model.buildConvolutionalLayer(2, 3, 1, 1, "zero-padding", "Orange Ball Photos all");
	/*    model.buildConvolutionalLayer(6, 6, 1, 1, 3, 1, 1, "zero-padding", "ImageText2");
		model.buildReluLayer(); 
		
		model.buildPoolingLayer(2, "MAX");

		model.buildHiddenLayer("LEAKYRELU"); 
		
		model.buildOutputLayer(2, "SOFTMAX", "BallTargets"); 
	*/
		
	//	model.buildInputLayer(400, 2, 400, "Inputs");
	
	
		model.buildInputLayerText(1953, 62, 1953, "Inputs", "Lookup");
		
	//	model.buildHiddenLayer(100, "LEAKYRELU");

		model.buildDropoutLayer(100, .5, "ELU");
		
		model.buildHiddenLayer(100, "LEAKYRELU");
	
		model.buildOutputLayer(2, "LINEAR");
	
		
//		trainer.train(model, 100, "ADAM"); 

		trainer.trainUntil(model, .77, 100, "ADAM");
		
	//	modelSaver.saveModel(model, trainer.weights); // for regular train
	
	
	
	
		modelSaver.saveModel(model, trainer.net.weights); // for train until 

	 
		


	//  modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
		
//		modelEvaluator.predictNCAA("basketBall81", "Lookup", "Connecticut", "South-Florida");
		
	//	modelEvaluator.predictNCAA(name, "Lookup", "North-Carolina-Central", "North-Dakota-St.");
	
	/*	
		modelEvaluator.predictNCAA(name, "Lookup", "VCU", "UCF");
		modelEvaluator.predictNCAA(name, "Lookup", "Mississippi-St.", "Liberty");
		modelEvaluator.predictNCAA(name, "Lookup", "Saint-Louis", "Virginia-Tech");
		modelEvaluator.predictNCAA(name, "Lookup", "Maryland", "Belmont");
		modelEvaluator.predictNCAA(name, "Lookup", "LSU", "Yale");
		modelEvaluator.predictNCAA(name, "Lookup", "Minnesota", "Louisville");
		modelEvaluator.predictNCAA(name, "Lookup", "Bradley", "Michigan-St.");
		modelEvaluator.predictNCAA(name, "Lookup", "Gonzaga", "Fairleigh-Dickinson");
		modelEvaluator.predictNCAA(name, "Lookup", "Syracuse", "Baylor");
		modelEvaluator.predictNCAA(name, "Lookup", "Marquette", "Murray-St.");
		modelEvaluator.predictNCAA(name, "Lookup", "Vermont", "Florida-St.");
		modelEvaluator.predictNCAA(name, "Lookup", "Texas-Tech", "Northern-Kentucky");
		modelEvaluator.predictNCAA(name, "Lookup", "Nevada", "Florida");
		modelEvaluator.predictNCAA(name, "Lookup", "Montana", "Michigan");
		modelEvaluator.predictNCAA(name, "Lookup", "Gardner-Webb", "Virginia");
		modelEvaluator.predictNCAA(name, "Lookup", "Mississippi", "Oklahoma");
		modelEvaluator.predictNCAA(name, "Lookup", "Oregon", "Wisconsin");
		modelEvaluator.predictNCAA(name, "Lookup", "Kansas-St.", "UC-Irvine");
		modelEvaluator.predictNCAA(name, "Lookup", "Saint-Mary's", "Villanova");
		modelEvaluator.predictNCAA(name, "Lookup", "Purdue", "Old-Dominion");
		modelEvaluator.predictNCAA(name, "Lookup", "Iowa", "Cincinnati");
		modelEvaluator.predictNCAA(name, "Lookup", "Tennessee", "Colgate");
		modelEvaluator.predictNCAA(name, "Lookup", "North-Carolina", "Iona");
		modelEvaluator.predictNCAA(name, "Lookup", "Washington", "Utah-St.");
		modelEvaluator.predictNCAA(name, "Lookup", "New-Mexico-St.", "Auburn");
		modelEvaluator.predictNCAA(name, "Lookup", "Northeastern", "Kansas");
		modelEvaluator.predictNCAA(name, "Lookup", "Ohio-St.", "Iowa-St.");
		modelEvaluator.predictNCAA(name, "Lookup", "Georgia-St.", "Houston");
		modelEvaluator.predictNCAA(name, "Lookup", "Seton-Hall", "Wofford");
		modelEvaluator.predictNCAA(name, "Lookup", "Abilene-Christian", "Kentucky");
		modelEvaluator.predictNCAA(name, "Lookup", "New-Mexico-St.", "Auburn");
	*/
		
		modelEvaluator.predictNCAA("cbb1", "Lookup", "Minnesota", "Penn-St.");
		
	}
}






