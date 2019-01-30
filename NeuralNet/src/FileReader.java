import java.io.File;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

public class FileReader {
	public int targetSize;
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
	
	public void readFileIntoList() { 
		while(scan.hasNextDouble()) {
			valuesFromFile.add(scan.nextDouble());
		}
	}

	public double[][] ListToArray(int dimension1, int dimension2) {
		int counter =0;
		int targetSize = determineTargetSize(dimension1, dimension2); 
		double[][] array = new double[dimension1][dimension2 + targetSize]; 
		
		for(int i=0; i < dimension1; i++) {
			for(int j=0; j < dimension2 + targetSize; j++) {
				array[i][j] = valuesFromFile.get(counter); 
				counter++; 
			}
		}
		return array; 
	}
	
	private List<double[][][]> ListToThreeDArray(int dimension1, int dimension2, int dimension3) {
		int counter =0;
		double[][][] array; 
		List<double[][][]> arrayList = new ArrayList<double[][][]>();
		
		for(int l=0; l<(valuesFromFile.size()/(dimension1*dimension2*dimension3)); l++) {
			array = new double[dimension1][dimension2][dimension3];
			for(int i=0; i < dimension1; i++) {
				for(int j=0; j < dimension2; j++) {
					for(int k=0; k < dimension3; k++) {
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
		int sizeOfInputs = dimension1*dimension2; 
		int targetArea = valuesFromFile.size() - sizeOfInputs;
		targetSize = (targetArea / dimension1);
		return (targetArea / dimension1);
	}
	
}
