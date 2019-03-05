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

public class FileReader {
	public int targetSize;
	String fileName = "";
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\";
	Scanner scan;
	BufferedReader read;
	ArrayList<Double> valuesFromFile = new ArrayList<Double>();
	ArrayList<String> stringList = new ArrayList<String>(); 
	ArrayList<Double> valuesFromFile2 = new ArrayList<Double>();
	ArrayList<String> stringList2 = new ArrayList<String>(); 
	

	public FileReader(String fileName) {
		this.fileName = fileName;
	}

	public double[][] readInputIntoArray(int dimension1, int dimension2) {
		initializeFileReader();
		readFileIntoList();
		return ListToArray(dimension1, dimension2);
	}
	
	public double[][] parseInputIntoArray(int dimension1, int dimension2, String lookup) {
		initializeBufferedReader(); 
		parseDataIntoLists(valuesFromFile, stringList); 
		initializeBufferedReader(lookup);
		parseDataIntoLists(valuesFromFile2, stringList2);
		
		buildLookupTable(); 
		return parseListsToArray(dimension1, dimension2); 
	}

	public List<double[][][]> readImageTextIntoList(int dimension1, int dimension2, int dimension3) {
		initializeFileReader();
		readFileIntoList();
		return ListToThreeDArray(dimension1, dimension2, dimension3);
	}
	
	


	public List<double[][][]> readImagesIntoList() {
		ImageReader imageReader = new ImageReader();
		return imageReader.readImageFile(fileName);
	}

	public void initializeFileReader() {
		try {
			File file = new File(fileName);
			scan = new Scanner(file);
		} catch (Exception e) {
			System.out.println("File not found");
			System.out.println(fileName);
		}
	}

	public void initializeBufferedReader() {
		try {
			read = new BufferedReader(new InputStreamReader(new FileInputStream(fileName)));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
	
	public void initializeBufferedReader(String lookup) {
		try {
			read = new BufferedReader(new InputStreamReader(new FileInputStream(strdFilePath + lookup + ".txt")));
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}

	public void readFileIntoList() {
		System.out.println("Slow file reader called");
		long start = System.nanoTime();
		
		while (scan.hasNextDouble()) {
			valuesFromFile.add(scan.nextDouble());
		}
		
		scan.close();
		
		long end = System.nanoTime();

		System.out.println("time" + (double) (end - start) / 1000000000);
	}

	public void readDataIntoList() {
		String line;
		double score;

		long start = System.nanoTime();

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

		
		long end = System.nanoTime();

		System.out.println("FileReader " + (double) (end - start) / 1000000000);
	}
	
	public void parseDataIntoLists(ArrayList<Double> valuesFromFile, ArrayList<String> stringList) {
		String line;
		String str; 
		double score;
		
		Pattern pattern = Pattern.compile(".*[a-zA-Z]+.*");
		Matcher matcher;

		long start = System.nanoTime();

		try {
			while ((line = read.readLine()) != null) {
				String[] Array = line.split(" ");
				for (String temp : Array) {
					if (!temp.isEmpty()) {
						matcher = pattern.matcher(temp);
						
						if(!matcher.matches()) {
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

		
		long end = System.nanoTime();

		System.out.println("FileReader " + (double) (end - start) / 1000000000);
	}
	
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
	
	private double[][] parseListsToArray(int dimension1, int dimension2) {
		int counter = 0;
		int counter2 = 0; 
		int counter3 = 0; 
		int targetSize = determineTargetSizeWithText(dimension1, dimension2);
		
		System.out.println(dimension1);
	
		double[] indvTeamStats; 
		double[][] array = new double[dimension1][dimension2 + targetSize];

		for(int i=0; i < dimension1; i++) {
			for(int j=0; j < stringList.size() / dimension1; j++) {
				indvTeamStats = textSearch(stringList.get(counter2));
				
				for(int k=0; k < indvTeamStats.length; k++) {
					array[i][counter] = indvTeamStats[k]; 
					counter++;
				} 
				counter2++; 	
			}
			for(int k=0; k<targetSize; k++) {
				array[i][counter] = valuesFromFile.get(counter3);
				counter++; 
				counter3++;
			}
			counter = 0; 
		}

		return array;
	}
	
	double[][] data; 
	private double[] textSearch(String team) {
		double[] stats = new double[data[0].length]; 
	
		for(int i=0; i<stringList2.size(); i++) {
			if(team.equals(stringList2.get(i))) {
				for(int j=0; j < data[0].length; j++) {
					stats[j] = data[i][j]; 
				}
			}
		}
		
		return stats;
	}
	
	private void buildLookupTable() {
		int counter = 0; 
		data = new double[stringList2.size()][(valuesFromFile2.size() / stringList2.size())]; 
		
		for(int i=0; i<data.length; i++) {
			for(int j=0; j < data[0].length; j++) {
				data[i][j] = valuesFromFile2.get(counter);
				counter++; 
			}
		}
		
	}
	

	private List<double[][][]> ListToThreeDArray(int dimension1, int dimension2, int dimension3) {
		int counter = 0;
		double[][][] array;
		List<double[][][]> arrayList = new ArrayList<double[][][]>();

		for (int l = 0; l < (valuesFromFile.size() / (dimension1 * dimension2 * dimension3)); l++) {
			array = new double[dimension1][dimension2][dimension3];
			for (int i = 0; i < dimension1; i++) {
				for (int j = 0; j < dimension2; j++) {
					for (int k = 0; k < dimension3; k++) {
						array[i][j][k] = valuesFromFile.get(counter);
						counter++;
					}
				}
			}
			arrayList.add(array);
		}
		return arrayList;
	}

	public int determineTargetSize(int dimension1, int dimension2) {
		int sizeOfInputs = dimension1 * dimension2;
		int targetArea = valuesFromFile.size() - sizeOfInputs;
		targetSize = (targetArea / dimension1);
		return (targetArea / dimension1);
	}
	
	public int determineTargetSizeWithText(int dimension1, int dimension2) {
		return valuesFromFile.size() / dimension1;
	}
	
}
