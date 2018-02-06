
public class BackPropagation {
	
	public void backPropagate() {
		
		double[][] result = Object.gs.getResult(); 
		double[][] marginOfError = Object.gs.AddAcross(Object.gs.getTarget(), result);
		double[][] var = Object.gs.ApplySigmoidDerivative(Object.gs.getResultPreSigmoid()); //result pre matrix
		double[][] deltaOutputSum = Object.gs.MultiplyAcross(marginOfError, var);
		Object.gs.setDeltaOutputSum(deltaOutputSum);
		
		double[][] ResultToHiddenWeightChange = Object.gs.MatrixMultiplication(deltaOutputSum, Object.gs.MatrixTranspose(Object.gs.getHiddenLayer3()));
		// setting randomWeights4 to new value, look at later in context of looping
		
		Object.gs.setWeights4(Object.gs.AddAcross(Object.gs.getWeights4(), ResultToHiddenWeightChange));
	
		double[][] test = Object.gs.MatrixMultiplication(Object.gs.MatrixTranspose(Object.gs.getWeights4()), deltaOutputSum); // [x, 1]*[1,y], result =  [x,y]
		// must adjust the multiplication method on variable choice to divide to each row in the matrix (for variable test)
		//specialized divide method to make deltaHiddenSum work 
		
		double[][] deltaHiddenSum = Object.gs.MultiplyAcross(test, Object.gs.ApplySigmoidDerivative(Object.gs.getHiddenLayer3PreSigmoid())); // derivative of hiddenNodePreSigmoid 
		 //set deltaHiddenSum
		
		//weights adjustment can be overridden with new data 
		double[][] weightAdjustment = Object.gs.MatrixMultiplication(Object.gs.MatrixTranspose(Object.gs.getHiddenLayer2()), deltaHiddenSum);
		System.out.println(java.util.Arrays.deepToString(weightAdjustment)); 
	}
}
