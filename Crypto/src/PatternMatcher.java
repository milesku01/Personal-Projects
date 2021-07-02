import java.util.List;
import java.math.RoundingMode;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;

public class PatternMatcher {
	String pattern; 
	
	public PatternMatcher() {
		
	}
	
	public void patternSearch(String pattern, double[][][] data) {
		
	}
	
	public void patternSearch(String pattern, double[] data) {
		if(pattern.equals("CHUNK")) {
			ChunkSearch chunkSearch = new ChunkSearch(data); 
			chunkSearch.locateChunks();
			//chunkSearch.simulateFutureTrading(data);
		}
	}
}


class PatternSearch {
	
}

class USHAPESearch extends PatternSearch{
	
}

class ChunkSearch extends PatternSearch {
	
	double[] data;
	
	public ChunkSearch(double[] data) {
		this.data = data;
	}
	
	void locateChunks() {
		
		int bestFitCheckSize = 8; 
		int percentCheckSize = 8; 
		int checkSize = 8; 
		double[] bestFitTemp = new double[bestFitCheckSize];
		double[] percentTemp = new double[percentCheckSize];
		double[] temp = new double[checkSize]; 
		double mean = 0; 
		double strdDev = 0; 
		double percentChange = 0; 
		
		double[] selectedData = reverseArray(getEveryNthIndex(data, 3));
		
		//double[] selectedData = reverseArray(data);
		
		double[] dataAdj = calculatePercentChanges(selectedData); 	
		List<Integer> strdDevIndexList = new ArrayList<Integer>(); 
		List<Integer> percentIndexList = new ArrayList<Integer>();
		
		List<Double> strdDevList = new ArrayList<Double>(); 
		List<Double> percentList = new ArrayList<Double>();
		
		List<Double> slopeList = new ArrayList<Double>();
		List<Integer> slopeIndexList = new ArrayList<Integer>();
		
		List<Integer> stabilityIndexList = new ArrayList<Integer>(); 
		
		for(int i=0; i < dataAdj.length-bestFitCheckSize; i++) { //search through every data point
			for(int j=0; j < bestFitCheckSize; j++) { //search within small subchunks of 10 hours, alter to search for important lengths of time
				bestFitTemp[j] = dataAdj[i+j]; //fills temp with the 10 data points
			}
			
			//System.out.println("Slopes " + calculateSlopeOfBestFit(bestFitTemp));
			
			if(Math.abs(calculateSlopeOfBestFit(bestFitTemp)) < .001) {
				slopeList.add(calculateSlopeOfBestFit(bestFitTemp));
				slopeIndexList.add(i);
				//System.out.println(i);
			}
		}
		
		//slopeIndexList = listIndexesOfRuns(slopeIndexList);
		
		for(int i=0; i < dataAdj.length - percentCheckSize; i++) { //search through every data point
			for(int j=0; j < percentCheckSize; j++) { //search within small subchunks of 5 hours, alter to search for important lengths of time
				//percentTemp[j] = dataAdj[i+j]; //fills temp with the 5 data points
				percentTemp[j] = selectedData[i+j];
			}
			
			//percentChange = calculateCompoundedIncrease(percentTemp); 
			
			percentChange = calculateEndIncrease(percentTemp); 
			--percentChange;
			percentList.add(percentChange);
			
			if(percentChange > .05) {
				percentIndexList.add(i);
			} 
		}
		
		  percentIndexList = listIndexesOfRuns(percentIndexList);
	
		
		for(int i=0; i < dataAdj.length-checkSize; i++) { //search through every data point
			for(int j=0; j < checkSize; j++) { //search within small subchunks of 10 hours, alter to search for important lengths of time
				temp[j] = dataAdj[i+j]; //fills temp with the 10 data points
			}
			
			mean = calculateMean(temp);
			strdDev = calculateStrdDev(temp, mean); 
			
			//System.out.println(mean); 
			
			if(Math.abs(strdDev) < .02) {
				strdDevIndexList.add(i); 
				strdDevList.add(strdDev);
			}
		}
		
		stabilityIndexList = getCommonIndexes(strdDevIndexList, slopeIndexList); 
		//stabilityIndexList = listIndexesOfRuns(stabilityIndexList); 
		
		
		//for(int i=0; i<indexList.size(); i++) {
			//System.out.println("Index " + indexList.get(i));
		//}
		
		//strdDevIndexList = listIndexesOfRuns(strdDevIndexList); 
		
		
		int counter = 0;
		int counter2 = 0;
		int counter3 = 0;
		int counter4 = 0;
		
		for(int i=0; i < dataAdj.length-checkSize; i++) { //may not work as well if checksizes differ
			// System.out.print(selectedData[i] + " ");
			 
			 if(counter < percentIndexList.size() && i == percentIndexList.get(counter)) {
				// System.out.print("Percent Increase " + percentList.get(percentIndexList.get(counter)) + " ");
				 counter++;
			 } 
			 
			 if(counter2 < strdDevIndexList.size() && i == strdDevIndexList.get(counter2)) {
				// System.out.print("Relativley Stable " + strdDevList.get(counter2) + " ");
				 counter2++; 
			 }
			 
			 if(counter3 < slopeIndexList.size() && i == slopeIndexList.get(counter3)) {
				// System.out.print("Slope Stable " + slopeList.get(counter3));
				 counter3++; 
			 }
			 
			 if(counter4 < stabilityIndexList.size() && i == stabilityIndexList.get(counter4)) {
				// System.out.print("Both " + stabilityIndexList.get(counter4));
				 counter4++; 
			 }
			// System.out.println();
		}
		
		//simulateBackTrading(stabilityIndexList, percentIndexList, selectedData); //simulates buying a selling using knowledge of the future to assess the efficeincy of the algorithm
		simulateFutureTrading(selectedData); 
	}
	
	private void simulateBackTrading(List<Integer> stability, List<Integer> priceIncrease, double[] prices) {
		double balance = 10000; 
	
		
		List<Double> buyOrders = new ArrayList<Double>(); 
		List<Double> buyOrderCost = new ArrayList<Double>(); 
		
		for(int i=0; i < prices.length; i++) {
			if(searchForTarget(priceIncrease, i) && balance > 0) { //price increase found, add buy order
				buyOrders.add(balance * 1); //setting buys at 20% of balance 
				buyOrderCost.add(prices[i]);
				balance *= 0; 
				
				System.out.print("Buy Orders ");
				displayList(buyOrders); 
				System.out.print("Buy Order Cost ");
				displayList(buyOrderCost);
				
			} else if(searchForTarget(stability, i)) {
				for(int j = 0; j < buyOrders.size(); j++) {
					if(buyOrderCost.get(j) < prices[i]) {
						System.out.println("Sell price " + prices[i+5]);
						balance += (buyOrders.get(j) * (prices[i+5] / buyOrderCost.get(j)) ); //assume that sell occurs after price increase
						buyOrders.remove(j); 
						buyOrderCost.remove(j);
						j--; //buyOrder is removed so to check the next index j must be decremented 
						System.out.println("Balance " + balance);
					}
				}
			}
		}	
	}
	
	
	public void simulateFutureTrading(double[] prices) { //use knowledge of the past month to make trades
		int checkSize = 8;
		double percentWagered = 1;
		//int pastDataSize = 24; 
		//double[] pastData = new double[pastDataSize]; 
		double[] temp = new double[checkSize];
		double[] tempAdj = new double[checkSize];
		
		double balance = 10000; 
		
		double[] dataAdj = calculatePercentChanges(prices); 
		
		List<Double> buyOrders = new ArrayList<Double>(); 
		List<Double> buyOrderCost = new ArrayList<Double>(); 
		
		double mean;
		double strdDev;
		double percentIncrease;
		double slopeOfBestFit;
		
		/*
		List<Integer> strdDevIndexList = new ArrayList<Integer>(); 
		List<Integer> percentIndexList = new ArrayList<Integer>();
		
		List<Double> strdDevList = new ArrayList<Double>(); 
		List<Double> percentList = new ArrayList<Double>();
		
		List<Double> slopeList = new ArrayList<Double>();
		List<Integer> slopeIndexList = new ArrayList<Integer>();
		
		List<Integer> stabilityIndexList = new ArrayList<Integer>(); 
	*/
	//	System.out.println("HERE " + prices[0]);
		
		for(int i=0; i<checkSize+1; i++) {
			System.out.println("Prices " + prices[i]);
		}
		
		for(int i=1; i < prices.length - checkSize; i++) { // calculate using 3 days of previous knowledge //ignore first price point for testing only
			for(int j=0; j < checkSize; j++) {
				temp[j] = prices[i+j];
			}
			for(int j=0; j < checkSize; j++) {
				tempAdj[j] = dataAdj[i+j]; 
			}
			
			mean = calculateMean(tempAdj);
			strdDev = calculateStrdDev(tempAdj, mean);
			
			percentIncrease = calculateEndIncrease(temp);
			percentIncrease--; 
			slopeOfBestFit = calculateSlopeOfBestFit(tempAdj); 
			
			System.out.print("Prices " + prices[i+checkSize] + " strdDev " + formatDecimal(strdDev) + " percentIncrease " + formatDecimal(percentIncrease) + " slopeOfBestFit " + formatDecimal(slopeOfBestFit) + "\n");
			
			if(buyCondition(balance, percentIncrease, prices[i+checkSize], buyOrderCost)) { //buy condition
				//buyOrder
				buyOrders.add(balance * percentWagered);  
				buyOrderCost.add(prices[i+checkSize]);
				balance *= (1-percentWagered); 
				
				System.out.print("Buy Orders ");
				displayList(buyOrders); 
				System.out.print("Buy Order Cost ");
				displayList(buyOrderCost);
			} 
			
			else if(sellCondition(slopeOfBestFit, strdDev, percentIncrease, prices[i+checkSize], buyOrderCost)) { //sell condition
				for(int j = 0; j < buyOrders.size(); j++) {
					if(1.10*buyOrderCost.get(j) < prices[i+checkSize]) {
						System.out.println("Sell price " + prices[i + checkSize]);
						balance += (buyOrders.get(j) * (prices[i + checkSize] / buyOrderCost.get(j)) ); //assume that sell occurs after price increase
						buyOrders.remove(j); 
						buyOrderCost.remove(j);
						j--; //buyOrder is removed so to check the next index j must be decremented 
						System.out.println("Balance " + balance);
					}
				}
			}
			
			double tempBalance = balance; 
			System.out.println();
			
			for(int j=0; j < buyOrders.size(); j++) {
				tempBalance += buyOrders.get(j);
			}
			
			System.out.println("Total Balance " + tempBalance);
			System.out.println("Total gains " + tempBalance / 10000); 
		}
		
		
		/*
		for(int i=0; i < prices.length - pastDataSize; i++) { // calculate using 3 days of previous knowledge
			for(int j=0; j < pastDataSize; j++) {
				pastData[j] = prices[i+j]; //3 days worth of data
			}
			
			for(int j=0; j < pastDataSize-checkSize; j++) {
				for(int k=0; k < checkSize; k++) {
					temp[k] = pastData[j+k]; //take samples from those three days
				}
				
				double mean = calculateMean(temp);
				double strdDev = calculateStrdDev(temp, mean);
				
				double percentIncrease = calculateEndIncrease(temp);
				double slopeOfBestFit = calculateSlopeOfBestFit(temp); 
				
				if() {
					
				}
				
				
				//System.out.println("Prices " + prices[j] + " strdDev " + formatDecimal(strdDev) + " percentIncrease " + formatDecimal(percentIncrease) + " slopeOfBestFit " + formatDecimal(slopeOfBestFit)); 
			}
		}
		*/
	}

	//buy condion: should be based on the lowest point (beginning / end) of a cycle
	//could be found by searching for location with sharp decrease and then subsequent increase 
	//
	private boolean buyCondition(double balance, double percentIncrease, double currentPrice, List<Double> buyOrders) {
		//return percentIncrease < -.05; 
		return (percentIncrease > .04 && balance > 0); 
	}
	
	private boolean sellCondition(double slopeOfBestFit, double strdDev, double percentIncrease, double currentPrice, List<Double> buyOrders) {
		//return percentIncrease > .05; 
		return (Math.abs(slopeOfBestFit) < 0.001 && strdDev < .05); //strd dev has less impact
	}
	
	private static void displayList(List<Double> percentIndexList) {
		for(int i=0; i < percentIndexList.size(); i++) {
			System.out.print(percentIndexList.get(i) + " ");
		}
		System.out.println(); 
	}
	
	private static double calculateSlopeOfBestFit(double[] inputs) { //built for equally spaced intervals
		double mean = calculateMean(inputs);
		double meanOfX = (inputs.length + 1) / 2.0; 
		double numerator = 0; 
		double denominator = 0;
		
		for(int i=1; i <= inputs.length; i++) { //makes calculation above easier
			numerator += ((inputs[i-1] - mean)*(i - meanOfX)); 
		}
		
		for(int i=1; i<=inputs.length; i++) { //probably a mathematical formula for this
			denominator += ((i - meanOfX)*(i - meanOfX)); 
		}
		return numerator / denominator; 
	}
	
	public static double calculateCompoundedIncrease(double[] inputs) {
		double percentChange = 1; 
		for (int i = 0; i < inputs.length; i++) {
			percentChange *= (1+inputs[i]);
		}
		
		return --percentChange; 
	}
	
	private static double calculateEndIncrease(double[] values) {
		return (values[values.length-1] / values[0]);
	}
	
	
	public static double calculateStrdDev(double[] inputs, double mean) {
		double strdDev = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			strdDev += Math.pow(inputs[i] - mean, 2); // changed
		}
		
		strdDev /= inputs.length;
		
		return Math.sqrt(strdDev);
	}

	public static double calculateMean(double[] inputs) {
		double runningTotal = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			runningTotal += inputs[i]; // location of the closing price
		}
		
		return runningTotal / inputs.length;
	}
	
	
	private static double[] calculatePercentChanges(double[] values){
		double[] percentChanges = new double[values.length]; 
		
		percentChanges[0] = 0; //this data point should not be accessed 
		
		for(int i=1; i < percentChanges.length; i++) {
			percentChanges[i] = (values[i] / values[i-1]) - 1.0; 
		}
		
		return percentChanges; 
	}
	
	private static double[] getEveryNthIndex(double[] data, int index) {
		int num = data.length / index; 
		double[] adjusted = new double[num];
		for(int i=0; i<num; i++) {
			adjusted[i] = data[index*i]; 
		}
		return adjusted; 
	}
	
	private List<Integer> listIndexesOfRuns(List<Integer> indexList) {
		int startOfStablePos = 0;
		List<Integer> list = new ArrayList<Integer>(); 
		
		for(int i=0; i < indexList.size() - 1; i++) {
			startOfStablePos = i; 
			
			while(indexList.get(i+1) == (1+indexList.get(i)) && (i+1) != indexList.size()-1) {
				i++; 
			}
			
			list.add(indexList.get(startOfStablePos)); 
			
			//System.out.println("STARTS " + slopeIndexList.get(startOfStablePos));
		}
		return list; 
	}
	
	public String formatDecimal(double n) {
		Double n1 = (Double) n; 
		DecimalFormat df = new DecimalFormat("#.#####");
		df.setRoundingMode(RoundingMode.CEILING);
		return df.format(n1.doubleValue());
	}
	
	private double[] reverseArray(double[] validData) {
		for(int i = 0; i < validData.length / 2; i++)
		{
		    double temp = validData[i];
		    validData[i] = validData[validData.length - i - 1];
		    validData[validData.length - i - 1] = temp;
		}
		return validData;
	}

	private List<Integer> getCommonIndexes(List<Integer> list1, List<Integer> list2) {
		List<Integer> common = new ArrayList<Integer>(); 
		
		for(int i=0; i < list1.size(); i++) {
			if(searchForTarget(list2, list1.get(i))) {
				common.add(list1.get(i));
			}
		}
		
		return common; 
	}
	
	private boolean searchForTarget(List<Integer> list, int target) {
		for(int i=0; i<list.size(); i++) { //can be made faster using while
			if(list.get(i) == target) {
				return true; 
			}
		}
		return false;
	}
}

