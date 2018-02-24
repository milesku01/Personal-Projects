
import java.io.BufferedOutputStream;
import java.io.DataOutputStream;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;

public class DeepLearningMain {

	static Scanner scan = new Scanner(System.in);

	public static void main(String[] args) throws IOException {

		int numofSets; // number of Sets
		int numofInput; // number of inputs in every set
		String WeightFilePath = "C:\\Users\\Miles\\Desktop\\Weights.txt";
		String LayersFilePath = "C:\\Users\\Miles\\Desktop\\HiddenLayers.txt"; 
		BufferedOutputStream bos; //for the weights
		BufferedOutputStream bosLayers; 
		
		
		
		System.out.println("Enter the number of sets of data");
		numofSets = scan.nextInt();
		System.out.println("Enter the number of datapoints per set");
		numofInput = scan.nextInt();
		
		int y = numofSets;
		int x = numofInput; 

		double[][] Inputs = new double[numofInput][numofSets];

		System.out.println("Enter all input");

		for (int j = 0; j < numofSets; j++) {
			for (int i = 0; i < numofInput; i++) {
				Inputs[i][j] = scan.nextDouble();
			}
		}
		Object.gs.setInputs(Object.gs.scalarMultiply(Inputs, .01));
		// x = y, obviously

		Object.gs.makeWeights(numofInput, numofSets); // creates the weights and
														// sets
		// them using
		// setters (within get/set
		// class)

		double[][] targetMatrix = new double[1][numofSets];

		System.out.println("Enter Targets");

		for (int i = 0; i < 1; i++) {
			for (int j = 0; j < numofSets; j++) {
				targetMatrix[i][j] = scan.nextDouble();
			}
		}
		Object.gs.setTarget(targetMatrix); // used in back propagation class

		ArrayList<Double> percentages = new ArrayList<Double>();
		percentages.add(0.0);

		int a;

		if (numofInput * numofSets <= 100) {
			a = 2500;
		} else {
			a = 10000;
		}

		for (int v = 0; v < a; v++) {
			System.out.println("Counter " + (v + 1));

			Object.fp.run(numofInput, numofSets);
			Object.bp.backPropagate();

			percentages.add(Object.gs.getCorrectness());
			// System.out.println(java.util.Arrays.deepToString(Object.gs.getResult()));
			if (v > 1000) {
				if (percentages.get(v + 1) < percentages.get(v)) {
					break;
				}

				if (v > 999 && percentages.get(v + 1) > percentages.get(v)) {
					if (percentages.get(v + 1) - percentages.get(v) > .003) {
						a++;
					}
				}

			}
		} // end of large for loop 
		
		try {
			
	   	bos = new BufferedOutputStream(new FileOutputStream(WeightFilePath)); 
		 
			
			for (int k = 0; k < y; k++) {
				for (int l = 0; l < x; l++) {
					bos.write((Object.gs.getWeights()[k][l] + "").getBytes()); 
					bos.write(System.lineSeparator().getBytes()); 
				}
				}
			for (int k = 0; k <x ; k++) {
				for (int l = 0; l <y ; l++) {
				bos.write((Object.gs.getWeightsInsert()[k][l] + "").getBytes());	
				bos.write(System.lineSeparator().getBytes()); 
				}
				}
			for (int k = 0; k < y; k++) {
				for (int l = 0; l < x; l++) {
					bos.write((Object.gs.getWeights2()[k][l] + "").getBytes());	
					bos.write(System.lineSeparator().getBytes()); 
				}
			}
			for (int k = 0; k < x; k++) {
				for (int l = 0; l < y; l++) {
					bos.write((Object.gs.getWeights3()[k][l] + "").getBytes());	
					bos.write(System.lineSeparator().getBytes()); 
				}
			}
			for (int k = 0; k < 1; k++) {
				for (int l = 0; l < x; l++) {
					bos.write((Object.gs.getWeights4()[k][l] + "").getBytes());	
					bos.write(System.lineSeparator().getBytes()); 
			
				}
			}
			
			bos.close(); 
		}
		catch(IOException e) {
			System.out.println("Exception concering the file writer "); 
		}
		
		try { 
			bosLayers = new BufferedOutputStream(new FileOutputStream(LayersFilePath));
			
			//come back here to know how data was taken out so it can be put back the same way
			for (int k = 0; k < Object.gs.getHiddenLayer().length; k++) { // equal to x
				for (int l = 0; l < Object.gs.getHiddenLayer()[0].length; l++) { //equal to y
					bosLayers.write((Object.gs.getHiddenLayer()[k][l] + "").getBytes()); 
					bosLayers.write(System.lineSeparator().getBytes()); 
				}
				}
			for (int k = 0; k < Object.gs.getHiddenLayerInsert().length; k++) { // equal to x
				for (int l = 0; l < Object.gs.getHiddenLayerInsert()[0].length; l++) { //equal to y
					bosLayers.write((Object.gs.getHiddenLayerInsert()[k][l] + "").getBytes()); 
					bosLayers.write(System.lineSeparator().getBytes()); 
				}
				}
			for (int k = 0; k < Object.gs.getHiddenLayer2().length; k++) { // equal to x
				for (int l = 0; l < Object.gs.getHiddenLayer2()[0].length; l++) { //equal to y
					bosLayers.write((Object.gs.getHiddenLayer2()[k][l] + "").getBytes()); 
					bosLayers.write(System.lineSeparator().getBytes()); 
				}
				}
			for (int k = 0; k < Object.gs.getHiddenLayer3().length; k++) { // equal to x
				for (int l = 0; l < Object.gs.getHiddenLayer3()[0].length; l++) { //equal to y
					bosLayers.write((Object.gs.getHiddenLayer3()[k][l] + "").getBytes()); 
					bosLayers.write(System.lineSeparator().getBytes()); 
				}
				}
			
			bosLayers.close();
		}
		catch (IOException e) {
			System.out.println("Problem with file writing");
		}
		

	}

}
