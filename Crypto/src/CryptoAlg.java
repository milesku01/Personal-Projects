
import java.util.*;


public class CryptoAlg {

	
	private static void searchForChunks(double[] values) {
		double[] percentChanges = calculatePercentChanges(values); 
		
		for(double number : values) {
		//for(double number : percentChanges) {
			System.out.println(number);
		}
	}

	private static void evaluationAlgorithm(String[] coins, double[][] coinValues) {
		double evaluation = 0;
		double[] coinEvals = new double[coins.length];

		for (int i = 0; i < coinValues.length; i++) {

			evaluation += (coinValues[i][4] / coinValues[i][9]);

			coinEvals[i] = evaluation;
			evaluation = 0;
		}

		mergeSort(coinEvals, coins, 0, coinEvals.length - 1);
		reverse(coins);
		reverse(coinEvals);
		display(coins, coinEvals);
	}

	private static void filterAlgorithm(String[] coins, double[][] coinValues) {
		int counter = 0;
		
		double[][] filteredCoins = new double[coinValues.length][3];
		
		ArrayList<String> microsList = new ArrayList<String>(); 
		ArrayList<Double> tradingVolumeToMarketCap = new ArrayList<Double>(); 
		
		// will have lots of wasted space

		for (int i = 0; i < filteredCoins.length; i++) {

			//if (coinValues[i][1] < 50000000) { // circulating supply
				//if (coinValues[i][9] < 250000 && coinValues[i][9] > 1) { //marketcap 
					if ((coinValues[i][4]) / (coinValues[i][9]) > .02) {

						microsList.add(coins[i]);
						filteredCoins[counter][0] = coinValues[i][1];
						filteredCoins[counter][1] = coinValues[i][9];
						filteredCoins[counter][2] = (coinValues[i][4]) / (coinValues[i][9]);
						tradingVolumeToMarketCap.add((coinValues[i][4]) / (coinValues[i][9]));
						counter++;
					}
				//}
			//}
		}
		
		double[] tradingVolumeToMarketCapArray = new double[tradingVolumeToMarketCap.size()]; 
		String[] micros = new String[microsList.size()]; 
		
		for(int i=0; i < tradingVolumeToMarketCap.size(); i++) {
			tradingVolumeToMarketCapArray[i] = tradingVolumeToMarketCap.get(i);
			micros[i] = microsList.get(i); 
		}
		
		mergeSort(tradingVolumeToMarketCapArray, micros, 0, micros.length - 1);
		
		//reverse(tradingVolumeToMarketCapArray);
		//reverse(micros);
		
		display(micros, tradingVolumeToMarketCapArray); 
		
		/*
		int counter2 = 0; 
		while((counter2 < filteredCoins.length) && (filteredCoins[counter2][0] != 0.0)) {
			System.out.print(micros[counter2] + " ");
			for(int i=0; i<filteredCoins[0].length; i++) {
				System.out.print(filteredCoins[counter2][i] + " " ); 
			}
			System.out.println();
			counter2++; 
		}
		
		*/
	}

	private static void strdDevAlgorithm(double[][][] historicalValues, String[] coins) {
		double[][][] percentChanges = calculatePercentChanges(historicalValues);

		double[] mean = calculateMean(percentChanges); //of closing data 

		double[] strdDev = calculateStrdDev(percentChanges, mean); //of closing data 

		// double[] adjustedStrdDev = normalizeStandardDevByPrice(strdDev,
		// historicalValues);

		mergeSort(strdDev, coins, 0, strdDev.length - 1);
		

		display(coins, strdDev);
	}

	
	private static double[] calculatePercentChanges(double[] values){
		double[] percentChanges = new double[values.length - 1]; 
		
		for(int i=0; i < percentChanges.length; i++) {
			percentChanges[i] = (values[i] / values[i+1]) - 1.0; 
		}
		
		return percentChanges; 
	}
	public static double[][][] calculatePercentChanges(double[][][] historicalData) {
		double[][][] percentChanges = new double[historicalData.length][historicalData[0].length
				- 1][historicalData[0][0].length];

		for (int i = 0; i < percentChanges.length; i++) {
			for (int j = 0; j < percentChanges[0].length; j++) {
				percentChanges[i][j][3] = ((historicalData[i][j][3] / historicalData[i][j + 1][3]) - 1.0);
			}
		}

		return percentChanges;
	}

	public static double[] normalizeStandardDevByPrice(double[] strdDev, double[][][] historicalData) {
		double[] adjustedStrdDev = new double[strdDev.length];

		for (int i = 0; i < strdDev.length; i++) {
			adjustedStrdDev[i] = (strdDev[i] / historicalData[i][0][3]);
		}
		return adjustedStrdDev;
	}

	public static double[] calculateStrdDev(double[][][] inputs, double[] mean) {

		double[] strdDev = new double[inputs.length];

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[0].length; j++) {
				strdDev[i] += Math.pow(inputs[i][j][3] - mean[i], 2); // changed
			}
		}
		for (int i = 0; i < inputs.length; i++) {
			strdDev[i] = strdDev[i] / inputs.length;
		}

		for (int i = 0; i < inputs.length; i++) {
			strdDev[i] = Math.sqrt(strdDev[i]);
		}
		return strdDev;
	}

	public static double[] calculateMean(double[][][] inputs) {

		double[] mean = new double[inputs.length];
		double runningTotal = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			for (int j = 0; j < inputs[0].length; j++) {
				runningTotal += inputs[i][j][3]; // location of the closing price
			}
			mean[i] = (runningTotal / inputs.length);
			runningTotal = 0.0;
		}
		return mean;
	}

	private static void display(String[] coins, double[] coinEvals) {
		for (int i = 0; i < coins.length; i++) {
			System.out.println(coins[i] + " " + coinEvals[i]);
		}
	}

	private static void reverse(double[] coinEvals) {
		Collections.reverse(Arrays.asList(coinEvals));
	}

	private static void reverse(String[] coinEvals) {
		Collections.reverse(Arrays.asList(coinEvals));
	}

	public static void mergeSort(double[] array, String[] array2, int left, int right) {
		if (right <= left)
			return;
		int mid = (left + right) / 2;
		mergeSort(array, array2, left, mid);
		mergeSort(array, array2, mid + 1, right);
		merge(array, array2, left, mid, right);
	}

	static void merge(double[] array, String[] array2, int left, int mid, int right) {
		// calculating lengths
		int lengthLeft = mid - left + 1;
		int lengthRight = right - mid;

		// creating temporary subarrays
		double leftArray[] = new double[lengthLeft];
		double rightArray[] = new double[lengthRight];

		String leftArray2[] = new String[lengthLeft];
		String rightArray2[] = new String[lengthRight];

		// copying our sorted subarrays into temporaries
		for (int i = 0; i < lengthLeft; i++) {
			leftArray[i] = array[left + i];
			leftArray2[i] = array2[left + i];
		}

		for (int i = 0; i < lengthRight; i++) {
			rightArray[i] = array[mid + i + 1];
			rightArray2[i] = array2[mid + i + 1];
		}

		// iterators containing current index of temp subarrays
		int leftIndex = 0;
		int rightIndex = 0;

		// copying from leftArray and rightArray back into array
		for (int i = left; i < right + 1; i++) {
			// if there are still uncopied elements in R and L, copy minimum of the two
			if (leftIndex < lengthLeft && rightIndex < lengthRight) {
				if (leftArray[leftIndex] < rightArray[rightIndex]) {
					array[i] = leftArray[leftIndex];
					array2[i] = leftArray2[leftIndex];
					leftIndex++;
				} else {
					array[i] = rightArray[rightIndex];
					array2[i] = rightArray2[rightIndex];
					rightIndex++;
				}
			}
			// if all the elements have been copied from rightArray, copy the rest of
			// leftArray
			else if (leftIndex < lengthLeft) {
				array[i] = leftArray[leftIndex];
				array2[i] = leftArray2[leftIndex];
				leftIndex++;
			}
			// if all the elements have been copied from leftArray, copy the rest of
			// rightArray
			else if (rightIndex < lengthRight) {
				array[i] = rightArray[rightIndex];
				array2[i] = rightArray2[rightIndex];
				rightIndex++;
			}
		}
	}
}

