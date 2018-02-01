import java.util.Scanner;


public class DeepLearningMain {
		
	static Scanner scan = new Scanner(System.in); 
	
	
	static ForwardPropagation fp = new ForwardPropagation(); 
	static GetSet gs = new GetSet(); 
	
	public static void main(String[] args) { 
	
		

		int numofSets; //number of Sets
		int numofInput;  //number of inputs in every set

		
		System.out.println("Enter the number of sets of data");
		numofSets = scan.nextInt();
		System.out.println("Enter the number of datapoints per set");
		numofInput = scan.nextInt();

		double[][] Inputs = new double[numofInput][numofSets];

		System.out.println("Enter all input");

		for (int i = 0; i < numofInput; i++) {
			for (int j = 0; j < numofSets; j++) {
				Inputs[i][j] = scan.nextDouble();
			}
		}
		gs.setInputs(Inputs); 
		// x = y, obviously
		
		gs.makeWeights(numofInput, numofSets); //creates the weights and sets them using 
												//setters (within get/set class)
		
		double[][] targetMatrix = new double[1][numofSets];

		System.out.println("Enter Targets");

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < numofSets; j++) {
				targetMatrix[i][j] = scan.nextDouble();
			}
		}
		gs.setTarget(targetMatrix); //used in back propagation class
		
		fp.run(numofInput, numofSets);  
		
	}
}
