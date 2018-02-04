
public class ForwardPropagation {

	

	void run(int x, int y) {

		
	
		

		// have to pass in the corresponding weights
		// also the result matrix is a special case and requires a different
		// method

		
		
		double[][] hiddenLayerOne = CreateNode(Object.gs.getInputs(), Object.gs.getWeights(), x, y); // pass
																							// in
																							// dimensions
																							// of
																							// array
																							// and
																							// then
																							// x,
																							// y
		Object.gs.setHiddenLayer(hiddenLayerOne); // result = [x,x]
		
	
		//hiddenLayerInsert is in between hiddenLayerOne and hiddenLayerTwo, and uses its own set of weights
		// change made to solve the deltaOutput sum problem
		
		double[][] hiddenLayerInsert = CreateNode(Object.gs.getHiddenLayer(), Object.gs.getWeightsInsert(), x, y); 
		Object.gs.setHiddenLayerInsert(hiddenLayerInsert); // result [x,y]
		
		double[][] hiddenLayerTwo = CreateNode(Object.gs.getHiddenLayerInsert(), Object.gs.getWeights2(), x, y); // result
																											// [x,x]
		Object.gs.setHiddenLayer2(hiddenLayerTwo);
		
		
		double[][] hiddenLayerThree = CreateNode(Object.gs.getHiddenLayer2(), Object.gs.getWeights3(),  x, y); // result
																											// [x,y]
		Object.gs.setHiddenLayer3(hiddenLayerThree);
		double[][] result = CreateNodeResult(Object.gs.getWeights4(), Object.gs.getHiddenLayer3(), x, y); // result
																										// =
																										// [1,y]
		Object.gs.setResult(result);

	}

	public double[][] CreateNode(double[][] Inputs, double[][] randomWeights,  int x,
			int y) {

		double[][] resultchange;
		double[][] resultchange2;

		resultchange = Object.gs.MatrixMultiplication(Inputs, randomWeights);
		resultchange2 = Object.gs.returnSigmoid(resultchange);

		return resultchange2;
	}

	public double[][] CreateNodeResult(double[][] randomWeights, double[][] Inputs,  int x,
			int y) {

		double[][] resultchange;
		double[][] resultchange2;

		resultchange = Object.gs.MatrixMultiplication(randomWeights, Inputs);
		resultchange2 = Object.gs.returnSigmoid(resultchange);
		
		return resultchange2;
	}
}
