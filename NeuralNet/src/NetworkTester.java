
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		DiagnosticTool dt = new DiagnosticTool();
		 
		String name = "worked2"; 
		
		dt.runStandardDiagnostic();
		
	
		
	//	modelEvaluator.checkModelAgainstActual(name, "match2", "Lookup");		
	
	
		model.buildInputLayerText(1953, 24, 1953, "WinLossGames", "Lookup3");
		
	//	model.buildInputLayerText(45, 24, 45, "match", "Lookup3"); 
		
		model.buildHiddenLayer(50, "TANH");
		
		model.buildOutputLayer(2, "SOFTMAX");
	
		
	//	trainer.train(model, 10, "ADAM"); 

		trainer.trainUntil(model, .78, 8, "ADAM");
		
	//	trainer.trainUntilMatchFound(model, 20 , "match2", "ADAM", "Lookup3");
		
	//	modelSaver.saveModel(model, trainer.weights); // for regular train
	
	
	//	modelSaver.saveModel(model, trainer.net.weights); // for train until 

	
	 
	 
	
		
	//	modelEvaluator.constructNCAABracket("basketBall76", "startingBracket", "Lookup"); 

	//  modelEvaluator.predict("NIC3", 4); 
		
	//	modelEvaluator.checkModelAgainstActual("basketBall762", "match", "Lookup3");
		
	//	modelEvaluator.predictNCAA("basketBall778", "Lookup3", "Connecticut", "South-Florida");
		
//	modelEvaluator.predictNCAA(name, "Lookup", "Auburn", "North-Carolina");
	
	
	}
}






