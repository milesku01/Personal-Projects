
public class BackPropagation {
	
	public void backPropagate() {
		double[][] result = Object.gs.getResult(); 
		double[][] marginOfError = Object.gs.SubtractAcross(Object.gs.getTarget(), result);
		double[][] var = Object.gs.ApplySigmoidDerivative(Object.gs.getResultPreSigmoid()); //result pre matrix
		double[][] deltaOutputSum = Object.gs.MultiplyAcross(marginOfError, var);
		Object.gs.setDeltaOutputSum(deltaOutputSum);
		
		double[][] ResultToHiddenWeightChange = Object.gs.MatrixMultiplication(deltaOutputSum, Object.gs.MatrixTranspose(Object.gs.getHiddenLayer3()));
		// setting randomWeights4 to new value, look at later in context of looping
		
		Object.gs.setWeights4(Object.gs.AddAcross(Object.gs.getWeights4(), ResultToHiddenWeightChange));
	
		double[][] deltaHiddenSum = Object.gs.MatrixMultiplication(Object.gs.MatrixMultiplication(deltaOutputSum, Object.gs.MatrixTranspose(Object.gs.getWeights4())), Object.gs.ApplySigmoidDerivative(Object.gs.getHiddenLayer3PreSigmoid())); // derivative of hiddenNodePreSigmoid 
	
	}
}
