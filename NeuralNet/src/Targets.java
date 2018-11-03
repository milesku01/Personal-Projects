
public class Targets {
	double[][] targets;

	public void determineTargets(double[][] layerValue, int numofInput) {
		int targetSize = layerValue[0].length - numofInput;
		targets = new double[layerValue.length][targetSize]; 
		
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < targetSize; j++) {
				targets[i][j] = layerValue[i][j + numofInput];
			}
		}
		
	}
	
}
