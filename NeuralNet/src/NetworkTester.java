
public class NetworkTester {
	static NetworkEvaluator modelEvaluator = new NetworkEvaluator();
	static ModelSaver modelSaver = new ModelSaver();
	
	public static void main(String[] args) {
		NetworkModel model = new NetworkModel();
		NetworkTrainer trainer = new NetworkTrainer(); 
		DiagnosticTool dt = new DiagnosticTool();
		 
		String name = "worked2"; 
		
	//	dt.runStandardDiagnostic();
		
		
	//	modelEvaluator.predict("TestModelXOR", 100, 0);
		
	//	modelEvaluator.predict("TestModelXOR", "TestFile");
		
	//	modelEvaluator.checkModelAgainstActual(name, "match2", "Lookup");		
	//	modelEvaluator.predictNCAAMatchup("Basketball752", "Lookup", "Minnesota", "Iowa");
	
		modelEvaluator.constructNCAABracket("Basketball7525", "startingBracket", "Lookup");
	
		
	//	modelEvaluator.checkModelAgainstActual("Basketball7525", "match2", "Lookup");
		
	//	model.buildInputLayer(4, 2, 4, "TestFile");
		
	//	model.buildHiddenLayer(4, "TANH");
		
	//	model.buildOutputLayer(1, "SIGMOID");
		
		
		
		
	//	model.buildInputLayerText(1953, 62, 1953, "WinLossGames", "Lookup");
		
		//model.buildInputLayerText(45, 24, 45, "match", "Lookup3"); 
		
	//	model.buildHiddenLayer(25, "RELU");
		
	//	model.buildHiddenLayer(10, "RELU");
		
		
	//	model.buildOutputLayer(2, "SOFTMAX");
	
		
	//	trainer.train(model, 20, "ADAM"); 

	//	trainer.trainUntil(model, .75, 12, "ADAM");
		
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






