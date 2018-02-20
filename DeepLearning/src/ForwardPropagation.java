
public class ForwardPropagation {

	

	void run(int x, int y) {

		
	
		

		// have to pass in the corresponding weights
		// also the result matrix is a special case and requires a different
		// method

		
		
		double[][] hiddenLayerOne = CreateNode(Object.gs.getInputs(), Object.gs.getWeights(), x, y); // pass
		double[][] hiddenLayerOnePreSigmoid = CreateNodePreSigmoid(Object.gs.getInputs(), Object.gs.getWeights(), x, y); 																					// in
		Object.gs.setHiddenLayerPreSigmoid(hiddenLayerOnePreSigmoid);																					// dimensions
																							// of
																							// array
																							// and
																							// then
																							// x,
																							// y
		Object.gs.setHiddenLayer(hiddenLayerOne); // result = [x,x]
		
	
		//hiddenLayerInsert is in between hiddenLayerOne and hiddenLayerTwo, and uses its own set of weights
		// change made to solve the deltaOutput sum problem
		double[][] hiddenLayerInsertPreSigmoid = CreateNodePreSigmoid(Object.gs.getHiddenLayer(), Object.gs.getWeightsInsert(), x, y);
		double[][] hiddenLayerInsert = CreateNode(Object.gs.getHiddenLayer(), Object.gs.getWeightsInsert(), x, y); 
		Object.gs.setHiddenLayerInsert(hiddenLayerInsert); // result [x,y]
		Object.gs.setHiddenLayerInsertPreSigmoid(hiddenLayerInsertPreSigmoid);
		double[][] hiddenLayerTwo = CreateNode(Object.gs.getHiddenLayerInsert(), Object.gs.getWeights2(), x, y); // result [x,x]
		double[][] hiddenLayerTwoPreSigmoid = CreateNodePreSigmoid(Object.gs.getHiddenLayerInsert(), Object.gs.getWeights2(), x, y); 
		
		Object.gs.setHiddenLayer2(hiddenLayerTwo);
		Object.gs.setHiddenLayer2PreSigmoid(hiddenLayerTwoPreSigmoid); 
		
		double[][] hiddenLayerThree = CreateNode(Object.gs.getHiddenLayer2(), Object.gs.getWeights3(),  x, y); // result
		double[][] hiddenLayerThreePreSigmoid= CreateNodePreSigmoid(Object.gs.getHiddenLayer2(), Object.gs.getWeights3(),  x, y); 																									// [x,y]
		Object.gs.setHiddenLayer3(hiddenLayerThree);
		Object.gs.setHiddenLayer3PreSigmoid(hiddenLayerThreePreSigmoid); 
		
		double[][] resultPreSigmoid = CreateNodeResultPreSigmoid(Object.gs.getWeights4(), Object.gs.getHiddenLayer3(), x, y); 
		double[][] result = CreateNodeResult(Object.gs.getWeights4(), Object.gs.getHiddenLayer3(), x, y); // result
																										// =
		Object.gs.setResultPreSigmoid(resultPreSigmoid); 																								// [1,y]
		Object.gs.setResult(result);
		
		double percentage = percentCorrect(result);
		System.out.println("Percent accurate " + percentage); 
		Object.gs.setCorrectness(percentage);
		
	}
	
	public double percentCorrect(double[][] A) {
		double percentCorrect =0;
		double avgPercentCorrect; 
	//	double[][] percentCorrectArray; 
		
		double[][] targets = Object.gs.getTarget(); //[1,y]
		double[][] result = Object.gs.getResult(); 
 		
		for(int i=0; i<targets[0].length; i++) { //targets[0].length = 'y'
			
			if(targets[0][i] > result[0][i] ) {
				percentCorrect+=(100*(result[0][i]/targets[0][i]));
			}
			
			else {
				percentCorrect+=(100*(targets[0][i])/result[0][i]);
			}
			
			
		}
		
		avgPercentCorrect = (percentCorrect / targets[0].length); 
		Object.gs.setCorrectness(avgPercentCorrect);
		
		return avgPercentCorrect; 
	}

	public double[][] CreateNode(double[][] Inputs, double[][] randomWeights,  int x,
			int y) {

		double[][] resultchange;
		double[][] resultchange2;

		resultchange = Object.gs.MatrixMultiplication(Inputs, randomWeights);
		resultchange2 = Object.gs.returnSigmoid(resultchange);

		return resultchange2;
	}
	
	public double[][] CreateNodePreSigmoid(double[][] Inputs, double[][] randomWeights,  int x,
			int y) {

		double[][] resultchange;

		resultchange = Object.gs.MatrixMultiplication(Inputs, randomWeights);
		

		return resultchange;
	}

	public double[][] CreateNodeResult(double[][] randomWeights, double[][] Inputs,  int x,
			int y) {

		double[][] resultchange;
		double[][] resultchange2;

		resultchange = Object.gs.MatrixMultiplication(randomWeights, Inputs);
		resultchange2 = Object.gs.returnSigmoid(resultchange);
		
		return resultchange2;
	}
	
	public double[][] CreateNodeResultPreSigmoid(double[][] randomWeights, double[][] Inputs,  int x,
			int y) {

		double[][] resultchange;

		resultchange = Object.gs.MatrixMultiplication(randomWeights, Inputs);
		
		
		return resultchange;
	}
}
