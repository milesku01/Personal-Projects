import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * class File reader is used to create file reader objects and read files and format the 
 * data into useful types
 */
public class FileReader {
	public int targetSize;
	String fileName = "";
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\NeuralNetworkRelated\\";
	Scanner scan;
	BufferedReader read;
	ArrayList<Double> valuesFromFile = new ArrayList<Double>();
	ArrayList<String> stringList = new ArrayList<String>();
	ArrayList<Double> valuesFromFile2 = new ArrayList<Double>();
	ArrayList<String> stringList2 = new ArrayList<String>();

	/**
	 * creates a fileReader object that reads from a file "fileName" 
	 * @param fileName
	 */
	public FileReader(String fileName) {
		this.fileName = fileName;
	}

	/**
	 * reads the data from the file into an array of size dimension1 by dimension2
	 * first creates the file reader and then reads all the data into a list via a scanner
	 * 
	 * @param dimension1 : fist dimension of the array 
	 * @param dimension2 : second dimension of the array 
	 * @return returns the formatted array 
	 */
	 public double[][] readInputIntoArray(int dimension1, int dimension2) {
		initializeFileReader();
		readDoublesFromFileIntoList();
		return ListToArray(dimension1, dimension2);
	} 
	
	/**
	 * reads the data from the file into a list of doubles
	 * first creates the file reader and then reads all the data into a list using a scanner  
	 * @return returns the array list of doubles
	 */
	public List<Double> readInputIntoList() {
		initializeFileReader();
		readDoublesFromFileIntoList();
		return valuesFromFile; 
	}

	/**
	 * parses the input from the file into arrays which are used to 
	 * first initialized buffered reader and then reads the information from the file that contains 
	 * the highly abstract information (ex. team1 team2 score). That information is then put into
	 * their respective lists (strings into a string list and doubles into a double list) 
	 * 
	 * Then, similarly the data is read from the lookup table file into lists - this is the less abstract 
	 * information i.e. (team stat1 stat2...) 
	 * 
	 * Then a lookup table is built in array format which can easily be used to compare teams. It formats
	 * the less abstract list into an array which can easily be accessed. 
	 * 
	 * Then finally the abstract lists are paired with the less abstract data to create a table of doubles
	 * which is the array that is returned. It replaces team names with their data. 
	 * 
	 * @param dimension1 integer that represents the number of games
	 * @param dimension2 integer that represents the total number of stats for both of the teams i.e. if 2 stats for each team then 4
	 * @param lookup the string used to identify the lookup table in the file system 
	 * @return 
	 */
	public double[][] parseInputIntoArray(int dimension1, int dimension2, String lookup) {
		initializeBufferedReader();
		parseDataIntoLists(valuesFromFile, stringList);
		initializeBufferedReader(lookup);
		parseDataIntoLists(valuesFromFile2, stringList2);

		buildLookupTable();
		return parseListsToArray(dimension1, dimension2);
	}

	/**
	 * Very simple method that initializes a scanner to a certain file
	 * However scanners are relatively slow 
	 */
	public void initializeFileReader() {
		try {
			File file = new File(fileName);
			scan = new Scanner(file);
		} catch (Exception e) {
			System.out.println("File not found");
			System.out.println(fileName);
		}
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
	 * Very simple method that initializes a buffered reader with a specific lookup String used
	 * when reading more than one file so the buffered reader may be specified
	 * @param lookup string that points the buffered reader to a specific file 
	 */
	public void initializeBufferedReader(String lookup) {
		try {
			read = new BufferedReader(new InputStreamReader(new FileInputStream(strdFilePath + lookup + ".txt")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	
	/**
	 * Uses a scanner to read all the doubles from a file into the valuesFromFile list
	 * adds the doubles to a list parsing them with a scanner and then adding them to a file
	 */
	public void readDoublesFromFileIntoList() {
		while (scan.hasNextDouble()) {
			valuesFromFile.add(scan.nextDouble());
		}

		scan.close();
	}

	/**
	 * Uses the buffered reader to read doubles into a list, runs faster than the above method 
	 * While lines have not been checked, split that line using spaces as breaks into a string array which is then parsed
	 * into doubles and then adding those to lists
	 */
	public void readDataIntoList() {
		String line;
		double score;

		try {
			while ((line = read.readLine()) != null) {
				String[] Array = line.split(" ");
				for (String temp : Array) {
					if (!temp.isEmpty()) {
						score = Double.valueOf(temp);
						valuesFromFile.add(score);
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

	//static boolean TorF = true; //TODO can remove? 

	/**
	 * Simultaneously adds strings and doubles to their respective lists using references to lists in arguments 
	 * so that the method can be used in the same method with different buffered reader objects and add them to different lists
	 * 
	 * While lines have not been checked, split that line into a string array and then first check for characteristics of a name using a pattern matcher
	 * searching for letters. If a match is found then add that string to list of strings. IF it doesn't match the pattern then add it to a list of doubles 
	 * @param valuesFromFile a reference to a list of doubles (generally statistics)
	 * @param stringList a reference to list of strings (generally abstract concepts like team names) 
	 */
	public void parseDataIntoLists(ArrayList<Double> valuesFromFile, ArrayList<String> stringList) {
		String line;
		String str;
		double score;

		Pattern pattern = Pattern.compile(".*[a-zA-Z]+.*");
		Matcher matcher;

		try {
			while ((line = read.readLine()) != null) {
				String[] Array = line.split(" ");
				for (String temp : Array) {
					if (!temp.isEmpty()) {
						matcher = pattern.matcher(temp);

						if (!matcher.matches()) {
							score = Double.valueOf(temp);
							valuesFromFile.add(score);
						} else {
							str = String.valueOf(temp);
							stringList.add(str);
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

	/**
	 * Converts a list of doubles into an array with dimensions dimension 1 and 2 
	 * This method assumes that data not in the specified dimensions is a target and that information
	 * is placed into extra columns of size targetSize
	 * 
	 * Nested for loops read the list into an array sequentially using a counter
	 * 
	 * @param dimension1 the number of rows in the specified array 
	 * @param dimension2 the number of columns in the specified array (without the target)
	 * @return returns the newly formatted array 
	 */
	private double[][] ListToArray(int dimension1, int dimension2) {
		int counter = 0;
		int targetSize = determineTargetSize(dimension1, dimension2);
		double[][] array = new double[dimension1][dimension2 + targetSize];

		for (int i = 0; i < dimension1; i++) {
			for (int j = 0; j < dimension2 + targetSize; j++) {
				array[i][j] = valuesFromFile.get(counter);
				counter++;
			}
		}
		return array;
	}

	/**
	 * parses informations from the values and strings of the respective lists into team 
	 * statistics that can be used by the machine learning algorithm 
	 * 
	 * creates space for a full array be allocating for the number of rows (number of games) and 
	 * number of columns (twice the individual amount of stats for each team plus the target size) 
	 * uses a set of 3 for loops nested to loop through the "2" dimensions of the number of games and the number of 
	 * strings per game and then a third loop for the third dimension of individual stats per team 
	 * 
	 * then to add the targets (which already exist as doubles in the values from file) a simple for loop loops through an additional time per row to add the targets 
	 * to the array 
	 * 
	 * @param dimension1 the number of games in the array 
	 * @param dimension2 the number of stats from both teams as raw stats
	 * @return returns the formatted array of raw statistics 
	 */
	private double[][] parseListsToArray(int dimension1, int dimension2) {
		int counter = 0;
		int counter2 = 0;
		int counter3 = 0;
		int targetSize = determineTargetSizeWithText(dimension1);

		double[] indvTeamStats;
		double[][] array = new double[dimension1][dimension2 + targetSize];

		for (int i = 0; i < dimension1; i++) {
			for (int j = 0; j < stringList.size() / dimension1; j++) {
				indvTeamStats = textSearch(stringList.get(counter2));

				for (int k = 0; k < indvTeamStats.length; k++) {
					array[i][counter] = indvTeamStats[k];
					counter++;
				}
				counter2++;
			}
			for (int k = 0; k < targetSize; k++) {
				array[i][counter] = valuesFromFile.get(counter3);
				counter++;
				counter3++;
			}
			counter = 0;
		}

		return array;
	}
	
	

	double[][] data;

	/**
	 * Searches the look up table for the name of the string (stringList2) if found in the list then 
	 * the statistics associated with that team is added to the stats array from the data array (built from lookup table)
	 * and then is returned 
	 * 
	 * If TorF isn't flagged then the string wasn't found in the lookup table and the name is
	 * printed to the console for debugging 
	 * @param team the name of the team to be searched for
	 * @return returns the stats associated with "team" 
	 */
	public double[] textSearch(String team) {
		boolean TorF = false;
		double[] stats = new double[data[0].length];

		for (int i = 0; i < stringList2.size(); i++) {
			if (team.equals(stringList2.get(i))) {
				TorF = true;
				for (int j = 0; j < data[0].length; j++) {
					stats[j] = data[i][j];
				}
			}
		}

		if (!TorF) {
			System.out.println("String not found: " + team);
		}

		return stats;
	}

	
	/**
	 * Creates a lookup table from the lookup file that contains the stats associated with a team to be searched for
	 * uses nested for loops to add the data to a formatted data array from the values from file 
	 */
	public void buildLookupTable() {
		int counter = 0;
		data = new double[stringList2.size()][(valuesFromFile2.size() / stringList2.size())];

		for (int i = 0; i < data.length; i++) {
			for (int j = 0; j < data[0].length; j++) {
				data[i][j] = valuesFromFile2.get(counter);
				counter++;
			}
		}
	}

	/**
	 * determines the number of targets per row for a given input. Determines how much of the "area" is targets
	 * by subtracting the "area" of the inputs from the total values and then dividing by the number of rows
	 * @param dimension1 the number of rows
	 * @param dimension2 the expected number of columns 
	 * @return returns the amount of targets per row 
	 */
	public int determineTargetSize(int dimension1, int dimension2) {
		int sizeOfInputs = dimension1 * dimension2;
		int targetArea = valuesFromFile.size() - sizeOfInputs;
		targetSize = (targetArea / dimension1);
		return (targetArea / dimension1);
	}

	/**
	 * @param dimension1 the number of rows 
	 * @return returns the amount of targets per row by dividing the target area by the number of rows
	 */
	public int determineTargetSizeWithText(int dimension1) {
		return valuesFromFile.size() / dimension1;
	}

}
