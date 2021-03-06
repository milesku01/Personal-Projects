import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class Targets {
	int numofOutputNeuron; 
	double[][] targetValues;
	public int targetSize; 
	double[][] testTargets; 
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\"; 
	FileReader fr;

	public void determineTargets(double[][] layerValue, int numofInput) {
		targetValues = new double[layerValue.length][targetSize]; 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < targetSize; j++) {
				targetValues[i][j] = layerValue[i][j + numofInput];
			}
		}
		
	}
	
	public void determineTestTargets(double[][] layerValue, int numofInput, int offSet) {
		testTargets = new double[layerValue.length][targetSize]; 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < targetSize; j++) {
				testTargets[i][j] = layerValue[i][j + numofInput];
			}
		}

		
	}
	

	
	public void splitTargets(double[][] targets) {
		int numofSets = targets.length;
		int targetSize = targets[0].length; 
		int trainingSize = (int)(.7 * numofSets); 
		int testingSize = numofSets-trainingSize;
		double[][] trainingData = new double[trainingSize][targetSize];
		double[][] testData = new double[testingSize][targetSize];
		
		if(numofSets > 90) { 
			trainingData = new double[trainingSize][targetSize]; 
			testData = new double[testingSize][targetSize]; 
			
			for(int i=0; i<trainingSize; i++) {
				for(int j=0; j<targetSize; j++) {
					trainingData[i][j] = targets[i][j];
				}
			}
			for(int i=trainingSize; i<numofSets; i++) {
				for(int j=0; j<targetSize; j++) {
					testData[i-trainingSize][j] = targets[i][j]; 
				}
			}
			
			testTargets = testData;
			targetValues = trainingData; 
		}
	}
	
	private void shuffleTargets(double[][] targets, long seed) {
	
		double[] array; 
		List<double[]> list = new ArrayList<double[]>();
		
		for(int i=0; i<targets.length; i++) {
			array = new double[targets[0].length]; 
			for(int j=0; j<targets[0].length; j++) {
				array[j] = targets[i][j];
			}
			list.add(array);
		}
		
		Collections.shuffle(list, new Random(seed));
		
		for(int i=0; i<targets.length; i++) {
			for(int j=0; j<targets[0].length; j++) {
				targets[i][j] = list.get(i)[j]; 
			}
		}
		
		System.out.println(java.util.Arrays.deepToString(targets));
		
		targetValues = targets; 
	}
	
}
