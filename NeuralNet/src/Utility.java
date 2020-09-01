
public class Utility {
	
	public static int globalNumOfSetsThreshold = 90; 
	
	public static boolean testSplitThreshold(Layer layer) {
		return ((InputLayer)layer).numofSets >= globalNumOfSetsThreshold; 
	}
	
	public static double[][] copyArray(double[][] input) {
		double[][] copy = new double[input.length][input[0].length]; 
		for(int i = 0; i < input.length; i++) {
			for(int j = 0; j < input[0].length; j++) {
				copy[i][j] = input[i][j];
			}
		}
		return copy;
	}
	
	//TODO: check that copying by reference this way is okay, several appends dont refer to the layerValue
	public static double[][] appendBiasColumn(double[][] layer) {
		double[][] inputsWithBiases = new double[layer.length][layer[0].length + 1];

		for (int i = 0; i < layer.length; i++) {
			for (int j = 0; j < layer[0].length; j++) {
				inputsWithBiases[i][j] = layer[i][j];
			}
		}
		for (int i = 0; i < layer.length; i++) {
			inputsWithBiases[i][layer[0].length] = 1;
		}
		
		return inputsWithBiases; 
	}
	
	public static double[][] matrixMultiplication(double[][] A, double[][] B) {
		int aRows = A.length;
		int aColumns = A[0].length;
		int bRows = B.length;
		int bColumns = B[0].length;
		double[][] C = new double[aRows][bColumns];

		if (aColumns != bRows) {
			throw new IllegalArgumentException("A:col: " + aColumns + " did not match B:rows " + bRows + ".");
		}

		for (int i = 0; i < aRows; i++) { // aRow
			for (int j = 0; j < bColumns; j++) { // bColumn
				for (int k = 0; k < aColumns; k++) { // aColumn
					C[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		return C;
	}
	
	
}
