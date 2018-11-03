import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FileReader {
	String fileName = ""; 
	Scanner scan; 
	List<Double> valuesFromFile = new ArrayList<Double>();
	
	public FileReader(String fileName) { 
		this.fileName = fileName; 
	}
	
	public double[][] readInputIntoArray(int dimension1, int dimension2) {
		initializeFileReader(); 
		readFileIntoList(); 
		return ListToArray(dimension1, dimension2); 
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
	
	public void readFileIntoList() { 
		while(scan.hasNextDouble()) {
			valuesFromFile.add(scan.nextDouble());
		}
	}
	
	public double[][] ListToArray(int dimension1, int dimension2) {
		int counter =0;
		int targetSize = getTargetSize(dimension1, dimension2); 
		double[][] array = new double[dimension1][dimension2 + targetSize]; 
		
		for(int i=0; i < dimension1; i++) {
			for(int j=0; j < dimension2 + targetSize; j++) {
				array[i][j] = valuesFromFile.get(counter); 
				counter++; 
			}
		}
		return array; 
	}
	
	public int getTargetSize(int dimension1, int dimension2) {
		int sizeOfInputs = dimension1*dimension2; 
		int targetArea = valuesFromFile.size() - sizeOfInputs;
		return (targetArea / dimension1);
	}
	
}
