import java.io.BufferedOutputStream;
import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class GettersSetters {

	public static double weightBias = .1;
	static double[][] randWeights;
	static double[][] randWeights2;
	static double[][] resultWeights;
	static double[][] inputs;
	double[][] Targets;
	static double[][] layerOne;
	static double[][] layerTwo;
	static double[][] result;
	static double loss;
	static double lossBasic;
	static double maxVal; 
	static double minVal; 
	static double rangeVal;
	static double midRangeVal; 
	static double[] meanVal; 
	static double[] strdDevVal; 
	
	public void setWeights(double[][] a) {
		randWeights = a;
	}

	public double[][] getWeights() {
		return randWeights;
	}

	public void setWeights2(double[][] a) {
		randWeights2 = a;
	}

	public double[][] getWeights2() {
		return randWeights2;
	}

	public void setResultWeights(double[][] a) {
		resultWeights = a;
	}

	public double[][] getResultWeights() {
		return resultWeights;
	}

	public void setInputs(double[][] input) {
		inputs = input;
	}

	public void setTarget(double[][] targetMatrix) {
		this.Targets = targetMatrix;
	}

	public double[][] getInputs() {
		return inputs;
	}

	public double[][] getTarget() {
		return Targets;
	}

	public void setLayerOne(double[][] a) {
		layerOne = a;
	}

	public double[][] getLayerOne() {
		return layerOne;
	}

	public void setLayerTwo(double[][] a) {
		layerTwo = a;
	}

	public double[][] getLayerTwo() {
		return layerTwo;
	}

	public void setResult(double[][] a) {
		result = a;
	}

	public double[][] getResult() {
		return result;
	}

	public void setLoss(double a) {
		loss = a;
	}

	public double getLoss() {
		return loss;
	}

	public void setLossBasic(double a) {
		lossBasic = a;
	}

	public double getLossBasic() {
		return lossBasic;
	}
	 ///////////////////////////
	
	public void setMinValue(double a) {
		minVal = a;
	}

	public static double getMinValue() {
		return minVal;
	}
	public void setMaxValue(double a) {
		maxVal = a;
	}

	public static double getMaxValue() {
		return maxVal;
	}
	public void setMidRange(double a) {
		midRangeVal = a;
	}

	public static double getMidRange() {
		return midRangeVal;
	}
	public void setRange(double a) {
		rangeVal = a;
	}

	public static double getRange() {
		return rangeVal;
	}
	
	public void setMean(double[] a) {
		meanVal = a;
	}

	public static double[] getMean() {
		return meanVal;
	}

	public void setStrdDev(double[] a) {
		strdDevVal = a;
	}
	public static double[] getStrdDev() {
		return strdDevVal;
	}
	


	public void createWeights(int numofLayers, int numofLayers2, int numofInputs) {
		double[][] randomWeights = new double[numofInputs + 1][numofLayers]; // +1
																				// accounts
																				// for
																				// bias
																				// layer
		// weights between input and hidden node 1
		for (int k = 0; k < numofInputs; k++) {
			for (int l = 0; l < numofLayers; l++) {
				randomWeights[k][l] = ((double) ((Math.random() * 1) - .5));
			}
			for (int i = 0; i < numofLayers; i++) { // wrong
				randomWeights[numofInputs][i] = weightBias;
			} // adds bias to weights

			setWeights(randomWeights);

			double[][] randomWeights2 = new double[numofLayers + 1][numofLayers2];

			for (int i = 0; i < numofLayers; i++) {
				for (int j = 0; j < numofLayers2; j++) {
					randomWeights2[i][j] = ((double) ((Math.random() * 1) - .5));
				}
			}
			for (int i = 0; i < numofLayers2; i++) {
				randomWeights2[numofLayers][i] = weightBias;
			}
			setWeights2(randomWeights2);

			double[][] resultWeights = new double[numofLayers2 + 1][1];

			for (int i = 0; i < numofLayers2; i++) {
				for (int j = 0; j < 1; j++) {
					resultWeights[i][j] = ((double) ((Math.random() * 1) - .5));
				}
			}

			resultWeights[numofLayers2][0] = weightBias;

			setResultWeights(resultWeights);

		}
	}

	public double sumAllElements(double[][] A) {
		double sum = 0.0;

		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				sum += A[i][j];
			}
		}

		return sum;
	}

	public double[][] absoluteValAllElements(double[][] A) {

		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {
				A[i][j] = Math.abs(A[i][j]);
			}
		}
		return A;
	}

	public double[][] MatrixMultiplication(double[][] A, double[][] B) {

		int aRows = A.length;
		int aColumns = A[0].length;
		int bRows = B.length;
		int bColumns = B[0].length;

		if (aColumns != bRows) {
			throw new IllegalArgumentException("A:col: " + aColumns
					+ " did not match B:rows " + bRows + ".");
		}

		double[][] C = new double[aRows][bColumns];
		for (int i = 0; i < aRows; i++) {
			for (int j = 0; j < bColumns; j++) {
				C[i][j] = 0.00000;
			}
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

	// need to initialize
	public double[][] AddAcross(double[][] a, double[][] b) {

		double[][] result = new double[a.length][a[0].length];

		if (a.length != b.length || a[0].length != b[0].length) {
			throw new IllegalArgumentException(
					"Input did not match, arrays must be same dimensions; Subtraction Method");
		}

		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[i][j] = a[i][j] + b[i][j];
			}
		}

		return result;
	}

	public double[][] SubtractAcross(double[][] a, double[][] b) {

		double[][] result = new double[a.length][a[0].length];

		if (a.length != b.length || a[0].length != b[0].length) {
			throw new IllegalArgumentException(
					"Input did not match, arrays must be same dimensions; Subtraction Method");
		}

		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[i][j] = a[i][j] - b[i][j];
			}
		}

		return result;
	}

	public double[][] ApplyTangent(double[][] a) { // returns the
		// double[][]
		// with
		int x = a.length;
		int y = a[0].length; // sigmoid

		double[][] tangentResult = new double[x][y]; // a is the matrix

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				tangentResult[i][j] = java.lang.Math.tanh(a[i][j]);
			}
		}

		return tangentResult;
	}

	public double[][] ApplyInverseTangent(double[][] a) { // returns the
		// double[][]
		// with
		int x = a.length;
		int y = a[0].length; // sigmoid

		double[][] tangentResult = new double[x][y]; // a is the matrix

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				tangentResult[i][j] = InverseTangent(a[i][j]);
			}
		}

		return tangentResult;
	}

	public double InverseTangent(double a) {

		if (Math.abs(a) > 1) {
			throw new IllegalArgumentException(
					"You cannot input a value greater than one");
		}
		double result = .5 * Math.log((1 + a) / (1 - a));

		return result;
	}

	public double[][] addBiases(double[][] a) { // to inputs

		double[][] biasedArray = new double[a.length][a[0].length + 1];
		for (int k = 0; k < a.length; k++) {
			for (int l = 0; l < a[0].length; l++) {
				biasedArray[k][l] = a[k][l];
			}
		}
		for (int i = 0; i < a.length; i++) {
			biasedArray[i][a[0].length] = 1; // sets biases to one
		}
		return biasedArray;
	}

	public double[][] addWeightBiases(double[][] a) {
		double[][] biasedArray = new double[a.length + 1][a[0].length];
		for (int k = 0; k < a.length; k++) {
			for (int l = 0; l < a[0].length; l++) {
				biasedArray[k][l] = a[k][l];
			}
		}
		for (int i = 0; i < a[0].length; i++) {
			biasedArray[a.length][i] = .1; // sets biases to .one
		}
		return biasedArray;
	}

	public double[][] MultiplyAcross(double[][] a, double[][] b) {

		double[][] result = new double[a.length][a[0].length];

		if (a.length != b.length || a[0].length != b[0].length) {
			throw new IllegalArgumentException(
					"Input did not match, arrays must be same dimensions; Multiplication Method");
		}

		for (int i = 0; i < a.length; i++) {
			for (int j = 0; j < a[0].length; j++) {
				result[i][j] = a[i][j] * b[i][j];
			}
		}

		return result;
	}

	public double[][] MatrixTranspose(double[][] m) {
		double[][] temp = new double[m[0].length][m.length];
		for (int i = 0; i < m.length; i++)
			for (int j = 0; j < m[0].length; j++)
				temp[j][i] = m[i][j];
		return temp;
	}

	public double[][] scalarMultiply(double[][] A, double scalarVal) {
		double[][] var = new double[A.length][A[0].length];

		for (int i = 0; i < A.length; i++) {
			for (int j = 0; j < A[0].length; j++) {

				var[i][j] = scalarVal * A[i][j];

			}
		}

		return var;
	}

	public static double getMax(double[][] inputArray) {
		double maxValue = inputArray[0][0];
		for (int i = 0; i < inputArray.length; i++) {
			for (int h = 0; h < inputArray[0].length; h++) {
				if (inputArray[i][h] > maxValue) {
					maxValue = inputArray[i][h];
				}
			}
		}
		return maxValue;
	}

	// Method for getting the minimum value
	public static double getMin(double[][] inputArray) {
		double minValue = inputArray[0][0];
		for (int i = 0; i < inputArray.length; i++) {
			for (int h = 0; h < inputArray[0].length; h++) {
				if (inputArray[i][h] < minValue) {
					minValue = inputArray[i][h];
				}
			}
		}
		return minValue;
	}

	public double[][] minMax(double[][] A) {
		double min = getMin(A);
		double max = getMax(A);
		
		setMinValue(min);
		setMaxValue(max); 
		
		double[][] result = A;

		for (int i = 0; i < A.length; i++) {
			for (int h = 0; h < A[0].length; h++) {
				result[i][h] = ((A[i][h] - min) / (max - min)) + 0.0;
			}
		}
		return result;

	}

	public double[][] normalize(double[][] A, int numofSets, int numofInput, double[] meanGet, 
			double[] strdDevGet, boolean TorF){

		double[][] result = new double[A.length][A[0].length];
		double[] mean = new double[numofInput];
		double[] strdDev = new double[numofInput]; 
	
		if(TorF == false) {
			
			for(int i =0; i< numofInput; i++) {
				for(int j=0; j< numofSets; j++){
					mean[i] += A[j][i]; 
				}
			}

			for(int i=0; i<numofInput; i++) {
				mean[i] = mean[i]/(double)numofSets; }
			
			//FIX BINARY ENCODING PROBLEM 
			
			for(int i =0; i< numofInput; i++) {
				for(int j=0; j< numofSets; j++){
					strdDev[i] = Math.pow(A[j][i]-mean[i], 2); 
				}
			}
			for(int i =0; i<numofInput; i++){
				strdDev[i] = strdDev[i]/(double)numofSets; 
			}
			for(int i =0; i< numofInput; i++){
				strdDev[i] = Math.sqrt(strdDev[i]); 
			}
			
		} else {
			mean = meanGet; 
			strdDev = strdDevGet; 
		}
		setStrdDev(strdDev);
		setMean(mean);
		
		
		for(int i =0; i< numofInput; i++) {
			for(int j=0; j< numofSets; j++){
				result[j][i] = (A[j][i] - mean[i])/strdDev[i]; 
			}
		}
		
		System.out.println("strdDev" + java.util.Arrays.toString(strdDev));
		System.out.println("mean" + java.util.Arrays.toString(mean));
		 

		return result;
	}

	public double[][] rangeNormalize(double[][] A) {

		 double midRange = ((getMax(A) + getMin(A)) / 2.0);
		 double range = ((getMax(A) - getMin(A)) / 2.0);
		 
		 setMidRange(midRange);
		 setRange(range); 

		for (int i = 0; i < A.length; i++) {
			for (int h = 0; h < A[0].length; h++) {
				A[i][h] = ((A[i][h] - midRange) / range);
			}
		}

		return A;
	}
	
	public double[][] InverseMinMax(double[][] A, double max, double min){ 
		
		for(int i =0; i<A.length;i++){
			for(int j=0; j<A[0].length; j++){
				A[i][j] = (A[i][j]*(max-min))+min; 
			}
		}
		
		return A; 
	}
	public double[][] InverseRangeNormalize(double[][] A, double range, double midRange){
		
		for(int i=0; i<A.length; i++) {
			for(int j=0; j<A[0].length; j++){
				A[i][j] = (A[i][j]*(range))+midRange; 
			}
		}
		
		return A; 
	}

}
