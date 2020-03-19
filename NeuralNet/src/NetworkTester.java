
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		DiagnosticTool dt = new DiagnosticTool();
		 
		String name = "worked2"; 
		
	//	dt.runStandardDiagnostic();
		
	
		
	//	modelEvaluator.checkModelAgainstActual(name, "match2", "Lookup");		
	
	/*
		model.buildInputLayer(26, 1, 26, "NIC");
	
	//	model.buildInputLayerText(1953, 24, 1953, "WinLossGames", "Lookup3");
		
	//	model.buildInputLayerText(45, 24, 45, "match", "Lookup3"); 
		
		model.buildHiddenLayer(1000, "RELU");
		
	//	model.buildHiddenLayer(100, "RELU");
		
		
	//	model.buildHiddenLayer(30, "RELU"); 
	
		model.buildOutputLayer(1, "LINEAR");
	
		
		trainer.train(model, 10000, "ADAM"); 

	//	trainer.trainUntil(model, .78, 10, "ADAM");
		
	//	trainer.trainUntilMatchFound(model, 20 , "match2", "ADAM", "Lookup3");
		
		modelSaver.saveModel(model, trainer.weights); // for regular train
	*/
	
	//	modelSaver.saveModel(model, trainer.net.weights); // for train until 

	
	 
	 
	
		
		modelEvaluator.constructNCAABracket("basketBall76", "startingBracket", "Lookup"); 

	//  modelEvaluator.predict("NIC3", 4); 
		
	//	modelEvaluator.checkModelAgainstActual("basketBall762", "match", "Lookup3");
		
	//	modelEvaluator.predictNCAA("basketBall778", "Lookup3", "Connecticut", "South-Florida");
		
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






