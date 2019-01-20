
public class Targets {
	int numofOutputNeuron; 
	double[][] targets;
	public int targetSize; 
	double[][] testTargets; 
	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\"; 
	FileReader fr;

	public void determineTargets(double[][] layerValue, int numofInput) {
		targets = new double[layerValue.length][targetSize]; 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < targetSize; j++) {
				targets[i][j] = layerValue[i][j + numofInput];
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
	
	public void determineConvolutionalTargets(int numofSets, int targetSize, String targetFile) {
		fr = new FileReader(strdFilePath + targetFile + ".txt");
		targets = fr.readInputIntoArray(numofSets, targetSize);
		splitTargets(targets);
	}
	
	public void splitTargets(double[][] targets) {
		int numofSets = targets.length;
		int targetSize = targets[0].length; 
		int trainingSize = (int)(.7 * numofSets); 
		int testingSize = numofSets-trainingSize;
		double[][] trainingData = new double[trainingSize][targetSize];
		double[][] testData = new double[testingSize][targetSize];
		
		if(numofSets > 140) { //roughly 70% of 140 is 100
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
			
			targets = trainingData; 
		}
	}
	
}
