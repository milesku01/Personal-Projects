
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		 
		String name = "worked2"; 
		
		
	//	modelEvaluator.checkModelAgainstActual(name, "match2", "Lookup");		
		
	//	model.buildInputLayer(4, 2, 4, "Inputs2");
	
	
	//	model.buildInputLayerText(1953, 62, 200, "WinLossGames", "Lookup");
	
		//hey checking
		
	//	model.buildInputLayerText(45, 62, 45, "match", "Lookup"); 
		
	//	model.buildHiddenLayer(80, "LEAKYRELU");
		
	//	model.buildHiddenLayer(100, "RELU");
		
	//	model.buildHiddenLayer(80, "LEAKYRELU"); 
	
		
	//	model.buildOutputLayer(2, "SOFTMAX");
	
		
	//	trainer.train(model, 15, "ADAM"); 

	//	trainer.trainUntil(model, .80, 20, "ADAM");
		
	//	trainer.trainUntilMatchFound(model, 12 , "match2", "ADAM", "Lookup");
		
	//	modelSaver.saveModel(model, trainer.weights); // for regular train
	
	
	//	modelSaver.saveModel(model, trainer.net.weights); // for train until 

	
	 
	 
	
		
		modelEvaluator.constructNCAABracket("basketBall72", "startingBracket", "Lookup"); 

	//  modelEvaluator.predict("basketball", 15, 32.8, 117.9, 85.1, 12, 23.66, 114.8, 91.2); 
		
		
//		modelEvaluator.predictNCAA("basketBall81", "Lookup", "Connecticut", "South-Florida");
		
//	modelEvaluator.predictNCAA(name, "Lookup", "Auburn", "North-Carolina");
	
	/*
	
		modelEvaluator.predictNCAA(name, "Lookup", "UCF", "VCU");
		modelEvaluator.predictNCAA(name, "Lookup", "Mississippi", "Liberty");
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
		modelEvaluator.predictNCAA(name, "Lookup", "Florida", "Nevada");
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
	*/
	
		
	//	modelEvaluator.predictNCAA(name, "Lookup", "Gonzaga", "Florida-St.");
	//	modelEvaluator.predictNCAA(name, "Lookup", "Purdue", "Saint-Mary's");
	//	modelEvaluator.predictNCAA(name, "Lookup", "Belmont", "Maryland");
	
	}
}






