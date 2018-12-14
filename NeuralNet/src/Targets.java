
public class Targets {
	double[][] targets;
	public int targetSize; 
	double[][] testTargets; 

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
	
}
