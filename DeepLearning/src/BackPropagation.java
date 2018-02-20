
public class BackPropagation {
	
	public void backPropagate() {
		
		double learningRate = 0.6;
		double[][] result = Object.gs.getResult(); 
		double[][] marginOfError = Object.gs.AddAcross(Object.gs.getTarget(), result);
		double[][] var = Object.gs.ApplySigmoidDerivative(Object.gs.getResultPreSigmoid()); //result pre matrix
		double[][] deltaOutputSum = Object.gs.MultiplyAcross(marginOfError, var);
		Object.gs.setDeltaOutputSum(deltaOutputSum);
		
		double[][] ResultToHiddenWeightChange = Object.gs.scalarMultiply(Object.gs.MatrixMultiplication(deltaOutputSum, Object.gs.MatrixTranspose(Object.gs.getHiddenLayer3())), learningRate);
		// setting randomWeights4 to new value, look at later in context of looping
		
		Object.gs.setWeights4(Object.gs.AddAcross(Object.gs.getWeights4(), Object.gs.scalarMultiply(ResultToHiddenWeightChange, -1))); // add negative
//		System.out.println(java.util.Arrays.deepToString(Object.gs.getWeights4()) + "weights, result to previous layer");
		
		double[][] test = Object.gs.MatrixMultiplication(Object.gs.MatrixTranspose(Object.gs.getWeights4()), deltaOutputSum); // [x, 1]*[1,y], result =  [x,y]
		// must adjust the multiplication method on variable choice to divide to each row in the matrix (for variable test)
		//specialized divide method to make deltaHiddenSum work 
		
		double[][] deltaHiddenSum = Object.gs.MultiplyAcross(test, Object.gs.ApplySigmoidDerivative(Object.gs.getHiddenLayer3PreSigmoid())); // derivative of hiddenNodePreSigmoid 
		 //set deltaHiddenSum
		
		//weights adjustment can be overridden with new data 
		double[][] weightAdjustment = Object.gs.scalarMultiply(Object.gs.MatrixMultiplication(Object.gs.MatrixTranspose(Object.gs.getHiddenLayer2()), deltaHiddenSum), learningRate);
		
		//System.out.println(java.util.Arrays.deepToString(weightAdjustment)); 
		
		double[][] newWeights = Object.gs.AddAcross(Object.gs.getWeights3(), Object.gs.scalarMultiply(weightAdjustment, -1));
		
	//	System.out.println(java.util.Arrays.deepToString(newWeights) + "new weights");
		
		//////////////////////////////////
		///////////////////////////////
		/////////////////////////////////
		
		//weights3 = [x][y]
		
		double[][] test2= Object.gs.MatrixMultiplication(deltaOutputSum, Object.gs.MatrixTranspose(Object.gs.getWeights3())); // [1,y]*[y,x], result =  [1,x]
		
		
		// must adjust the multiplication method on variable choice to divide to each row in the matrix (for variable test)
		//specialized divide method to make deltaHiddenSum work 
		
		double[][] deltaHiddenSum2 = Object.gs.specializedMultiplyAcross(test2, Object.gs.ApplySigmoidDerivative(Object.gs.getHiddenLayer2PreSigmoid())); // derivative of hiddenNodePreSigmoid 
		// [1,x][x,x]  need specializedMultiply across  
		//result = [x,x]
		
		//weights adjustment can be overridden with new data 
		double[][] weightAdjustment2 = Object.gs.scalarMultiply(Object.gs.MatrixMultiplication( Object.gs.MatrixTranspose(Object.gs.getHiddenLayerInsertPreSigmoid()), deltaHiddenSum2), learningRate); //[y,x]*
		
		//System.out.println(java.util.Arrays.deepToString(weightAdjustment)); 
	
		
		double[][] newWeights2 = Object.gs.AddAcross(Object.gs.getWeights2(), Object.gs.scalarMultiply(weightAdjustment2, -1)); //weights2 = [y,x]
		
		//System.out.println(java.util.Arrays.deepToString(newWeights2) + "new weights 2");
		
		//////////////////////////
		//////////////////////////
		 
		//problem with output of this one [x,x,x,0.0] 
		
		double[][] test3 = Object.gs.specializedDivide(deltaOutputSum, Object.gs.getWeights2()); //[y,x] = result
		
		double[][] deltaHiddenSum3 = Object.gs.MultiplyAcross(test3, Object.gs.ApplySigmoidDerivative(Object.gs.MatrixTranspose(Object.gs.getHiddenLayerInsertPreSigmoid()))); //[y,x] =result
		
		double[][] weightAdjustment3 = Object.gs.scalarMultiply(Object.gs.MatrixMultiplication(deltaHiddenSum3, Object.gs.MatrixTranspose(Object.gs.getHiddenLayerPreSigmoid())), learningRate); //[y,x]
		
		double[][] newWeights3 = Object.gs.AddAcross(Object.gs.getWeightsInsert(), Object.gs.scalarMultiply(weightAdjustment, -1)); 
		
		//System.out.println(java.util.Arrays.deepToString(newWeights3) + "new weights3 ");
		
		//System.out.println(java.util.Arrays.deepToString(weightAdjustment3)); 
		
		////////////////////////
		/////////////////////////////
		
		double[][] test4 = Object.gs.MatrixMultiplication(deltaOutputSum,  Object.gs.MatrixTranspose(Object.gs.getWeightsInsert()));
		
		double[][] deltaHiddenSum4 = Object.gs.specializedMultiplyAcross(test4, Object.gs.ApplySigmoidDerivative(Object.gs.getHiddenLayerPreSigmoid())); 
	
		double[][] weightAdjustment4 = Object.gs.scalarMultiply(Object.gs.MatrixMultiplication(Object.gs.MatrixTranspose(Object.gs.getInputs()), deltaHiddenSum4), learningRate); 
		
		double[][] newWeights4 = Object.gs.AddAcross(Object.gs.getWeights() ,Object.gs.scalarMultiply(weightAdjustment4, -1)); 
		
		//System.out.println("Weight adjustment 4 " + java.util.Arrays.deepToString(weightAdjustment4));
		
//		System.out.println(java.util.Arrays.deepToString(newWeights4) + "new weights 4"); 
		Object.gs.setWeights3(newWeights);
		Object.gs.setWeights2(newWeights2);
		Object.gs.setWeightsInsert(newWeights3);
		Object.gs.setWeights(newWeights4);
		
		
		
		
		
		}
		
	
}
