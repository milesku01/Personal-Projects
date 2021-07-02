import java.io.BufferedReader;

import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class Parser {
	String fileName;
	BufferedReader read;
	ArrayList<Double> valuesFromFile = new ArrayList<Double>();
	ArrayList<String> stringList = new ArrayList<String>();

	/**
	 * creates a fileReader object that reads from a file "fileName" 
	 * @param fileName
	 */
	public Parser(String fileName) {
		this.fileName = fileName;
	}


	public int getCoinSize() {
		return stringList.size(); 
	}
	
	public int getNumOfValuesSize() {
		return valuesFromFile.size(); 
	}
	
	public void initialize() {
		initializeBufferedReader();
		parseDataIntoLists(valuesFromFile, stringList);
	}
	
	public void readFileIntoArray(String[] coins, double[][] coinValues,  int numOfMetrics) {
		ListToArray(coins, coinValues, numOfMetrics); 
	}
	
	public double[] readFileIntoArray() {
		return ListToArray();
	}
	
	public void readHistoricalFile(String[] coins, double[][][] historicalValues, int numOfMetrics, int numOfHistoricalPoints) {
		ListToArray(coins, historicalValues, numOfMetrics, numOfHistoricalPoints); 
	}
	
	public void readHistoricalBTCFile(double[] historicalValues, int numOfDesiredValues, int numOfHoursInBetween) {
		ListToArray(historicalValues, numOfDesiredValues, numOfHoursInBetween); 
	}
	
	/**
	 * Very simple method that initializes a buffered reader with the single file used for 
	 * inputs (fileName)
	 */
	public void initializeBufferedReader() {
		try {
			read = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	/**
	 * Simultaneously adds strings and doubles to their respective lists using references to lists in arguments 
	 * so that the method can be used in the same method with different buffered reader objects and add them to different lists
	 * 
	 * While lines have not been checked, split that line into a string array and then first check for characteristics of a name using a pattern matcher
	 * searching for letters. If a match is found then add that string to list of strings. IF it doesn't match the pattern then add it to a list of doubles 
	 * @param valuesFromFile a reference to a list of doubles (generally statistics)
	 * @param stringList a reference to list of strings (generally abstract concepts like team names) 
	 */
	private void parseDataIntoLists(ArrayList<Double> valuesFromFile, ArrayList<String> stringList) {
		String line;
		String str;
		double score;
		String regexDouble = "-?\\d+(\\.\\d+)?(E-?\\d+)?(E\\+?\\d+)?(E?\\d+)?(e-?\\d+)?(e\\+?\\d+)?(e?\\d+)?";

		//Pattern pattern = Pattern.compile(".*[a-zA-Z]+.*");
		Pattern pattern = Pattern.compile(regexDouble); 
		Matcher matcher;

		try {
			while ((line = read.readLine()) != null) {
				String[] Array = line.split(" ");
				for (String temp : Array) {
					if (!temp.isEmpty()) {
						matcher = pattern.matcher(temp);

						if(matcher.matches()) {
						//if (!matcher.matches()) {
							score = Double.valueOf(temp);
							valuesFromFile.add(score);
						} else {
							str = String.valueOf(temp);
							if(str.equals("None")) {
								valuesFromFile.add((double) 2100000000); //Arbitrarily large num  
							} else {
								stringList.add(str);
							}
						}
					}
				}
			}

			valuesFromFile.trimToSize();

			read.close();

		} catch (NumberFormatException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}
	
	private double[] ListToArray() {
		double[] data = new double[valuesFromFile.size()];
		for(int i=0; i<valuesFromFile.size(); i++) {
			data[i] = valuesFromFile.get(i); 
		}
		return data;
	}
	
	private void ListToArray(String[] coins, double[][] coinValues, int numOfMetrics) {
		int numOfCoins = stringList.size(); 
		
		System.out.println(numOfCoins + " " + valuesFromFile.size());
		
		for(int i=0; i<numOfCoins; i++) {
			coins[i] = stringList.get(i); 
		}
		
		int counter = 0;
		for (int i = 0; i < numOfCoins; i++) {
			for (int j = 0; j < numOfMetrics; j++) {
				coinValues[i][j] = valuesFromFile.get(counter);
				counter++;
			}
		}
	}
	
	private void ListToArray(String[] coins, double[][][] historicalValues, int numOfMetrics, int numOfHistoricalPoints) {
		int numOfCoins = stringList.size(); // numberOfCoins
	
		
		for(int i=0; i<numOfCoins; i++) {
			coins[i] = stringList.get(i); 
		}
		
		int counter = 0;
		for (int i = 0; i < numOfCoins; i++) {
			for (int j = 0; j < numOfHistoricalPoints; j++) {
				for(int k=0; k < numOfMetrics; k++) {
					historicalValues[i][j][k] = valuesFromFile.get(counter);
					counter++;
				}
			}
		}
	}
	
	private void ListToArray(double[] historicalValues, int numOfDesiredValues, int numOfHoursInBetween) { //can be broken by user
		for(int i=0; i<numOfDesiredValues; i++) {
			historicalValues[i] = valuesFromFile.get(numOfHoursInBetween * i); 
		}
	}

}
