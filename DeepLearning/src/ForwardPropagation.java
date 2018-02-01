
public class ForwardPropagation {

	private double[][] Inputs; 
	private double[][] targetMatrix;
	private double[][] result;

	
	 static GetSet gs = new GetSet(); 
	
	void run(int x, int y) { 
		
		//don't know if need this here 
		Inputs = gs.getInputs(); 
		targetMatrix = gs.getTarget(); 
		
		double[][] hiddenLayerOne = CreateNode(Inputs, x, y, y, x, x, y); // pass in dimensions of array and then x, y
		gs.setHiddenLayer(hiddenLayerOne);								 // result = [x,x]
		double[][] hiddenLayerTwo = CreateNode(gs.getHiddenLayer(), x, x, x, x, x, y); //result = [x,x]
		gs.setHiddenLayer2(hiddenLayerTwo);
		double[][] hiddenLayerThree = CreateNode(gs.getHiddenLayer2(), x, x, x, y, x, y); //result [x,y] 
		gs.setHiddenLayer3(hiddenLayerThree);
		double[][] result = CreateNode(gs.getHiddenLayer3(), 1, x, x, y, x, y); //result = [1,y]
		gs.setResult(result);
		
		
	}
	
	public double[][] CreateNode (double[][] Inputs, int a , int b, int c, int d, int x, int y) { 
		
		double[][] randomWeights = gs.getWeights(); 
		
		result = gs.MatrixMultiplication(Inputs, randomWeights, a, b, c, d); 
		this.result = gs.returnSigmoid(result, x, y);
		
		return result; 
	}
}
