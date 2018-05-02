import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Scanner;


public class NeuralNet {

	public static int numofNeuronLayerOne = 4; // subject to change
	public static int numofNeuronLayerTwo = 2;
	static Scanner scan = new Scanner(System.in);
	static Thread timer = new Thread();

	public static void main(String[] args) throws InterruptedException {
		int numofInput;
		int numofSets;
		//boolean TorF = false;
		int targetBoolean=0;
		String function;
		String targetNormalize;
		
		boolean TorF = false; 
		
		String WeightFilePath = "C:\\Users\\Miles\\Desktop\\Weights.txt";
		String LossFilePath = "C:\\Users\\Miles\\Desktop\\Loss.txt";
		ArrayList<Double> lossList = new ArrayList<Double>();

		BufferedOutputStream bos; // for the weights

		System.out
				.println("Would you like to create a function?  y or yes/ Anything else means make prediction");
		function = scan.nextLine();
		
		System.out.println("Normalize targets between [0,1]? y or yes/ Anything else [-1,1]");
		targetNormalize = scan.nextLine();
		
		if(targetNormalize.equals("y") || targetNormalize.equals("yes")) {
			targetBoolean = 1; // range normalization
		}else if (targetNormalize.equals("n") || targetNormalize.equals("no")){
			targetBoolean = 2; //no normalization
		}

		if (function.equalsIgnoreCase("y") || function.equalsIgnoreCase("yes")) {

			System.out.println("Enter the number of sets of data");
			numofSets = scan.nextInt();
			System.out.println("Enter the number of datapoints per set");
			numofInput = scan.nextInt();

			double[][] Inputs = new double[numofSets][numofInput];
			double[][] biasedInputs = new double[numofSets][numofInput + 1];
			double[][] biasedInputs2 = new double[numofSets][numofInput + 1]; // Necessary 
			double[][] targets = new double[numofSets][1];

		if(TorF == true) {
			System.out.println("Enter all input");

			for (int j = 0; j < numofSets; j++) {
				for (int i = 0; i < numofInput; i++) {
					Inputs[j][i] = scan.nextDouble();
				}
			} // gets inputs without biases
			
			System.out.println("Initial inputs " + java.util.Arrays.deepToString(Inputs));
			
			
			Inputs = Objects.gtst.normalize(Inputs, numofSets, numofInput);

			System.out.println(java.util.Arrays.deepToString(Inputs));
			
			for (int j = 0; j < numofSets; j++) {
				for (int i = 0; i < numofInput; i++) {
					biasedInputs[j][i] = Inputs[j][i];
				}
			} // gets inputs, STILL without biases

			for (int i = 0; i < numofSets; i++) {
				biasedInputs[i][numofInput] = 1; // sets biases to one
			}
			System.out.println("Enter targets ");

			for (int n = 0; n < numofSets; n++) {
				targets[n][0] = scan.nextDouble();
			}
		
		} else {
			
			
			double[][] Inputs2 = new double[numofSets][numofInput+1];
			
		
			System.out.println("Enter Inputs that contain targets");
			for (int j = 0; j < numofSets; j++) {
				for (int i = 0; i < numofInput+1; i++) {
					Inputs2[j][i] = scan.nextDouble();
				}
			} // gets inputs without biases
			
			//System.out.println("Initial inputs " + java.util.Arrays.deepToString(Inputs));
			
			for(int n=0; n<numofSets; n++) {
				targets[n][0] = Inputs2[n][numofInput]; 
			}
			Inputs2 = Objects.bdp.removeLastColumn(Inputs2); 
	//		Inputs2 = Objects.gtst.normalize(Inputs2, numofSets, numofInput);

			//System.out.println(java.util.Arrays.deepToString(Inputs));
			
			for (int j = 0; j < numofSets; j++) {
				for (int i = 0; i < numofInput; i++) {
					biasedInputs2[j][i] = Inputs2[j][i];
				}
			} // gets inputs, STILL without biases

			for (int i = 0; i < numofSets; i++) {
				biasedInputs2[i][numofInput] = 1; // sets biases to one
			}
			
		} //end of else 
			
			System.out.println("Old targets " +java.util.Arrays.deepToString(targets));

			int counter = 1;
			
			if(TorF == true) {
			Objects.gtst.setInputs(biasedInputs);
			} else {
				Objects.gtst.setInputs(biasedInputs2);
			}
			
			if(targetBoolean == 0) {
			Objects.gtst.setTarget(Objects.gtst.minMax(targets));
			} else if (targetBoolean == 1){ 
			Objects.gtst.setTarget(Objects.gtst.rangeNormalize(targets));
			} else {
			Objects.gtst.setTarget(targets);
			}

			Objects.gtst.createWeights(numofNeuronLayerOne,
					numofNeuronLayerTwo, numofInput);


			//System.out.println("Weights "
				//	+ java.util.Arrays.deepToString(Objects.gtst.getWeights()));

				System.out.println("New Targets " + java.util.Arrays.deepToString(Objects.gtst.getTarget()));
			
				int m = 5000;
			for (int i = 0; i < m; i++) {

				System.out.println("\n" + "Counter " + counter);
				counter++;
				Objects.fdp.CreateLayer(Objects.gtst.getInputs());
				Objects.fdp.CreateSecondLayer(Objects.gtst.getLayerOne());
				Objects.fdp.CreateResult(Objects.gtst.getLayerTwo());

			/*
				System.out.println("1 "
						+ java.util.Arrays.deepToString(Objects.gtst
								.getWeights()));
				System.out.println("2 "
						+ java.util.Arrays.deepToString(Objects.gtst
								.getWeights2()));
				System.out.println("3 "
						+ java.util.Arrays.deepToString(Objects.gtst
								.getResultWeights()));
								
			*/
			//	System.out.println("Targets "+ java.util.Arrays.deepToString(Objects.gtst.getTarget()));
				//System.out.println("Result "
					//	+ java.util.Arrays.deepToString(Objects.gtst
						//		.getResult()));

				if (i < m - 1) {
					Objects.bdp.runIteration(numofSets, counter);
				}

				System.out.println("Loss " + (Objects.gtst.getLoss()));
				System.out.println("Basic Loss " + Objects.gtst.getLossBasic());
				lossList.add(Objects.gtst.getLoss());

				Thread.sleep(0);

			}
			System.out.println(" \n Would you like to save these weights? ");
			function = scan.nextLine();
			function = scan.nextLine();

			if (function.equalsIgnoreCase("y")
					|| function.equalsIgnoreCase("yes")) {

				try { // //

					bos = new BufferedOutputStream(new FileOutputStream(
							LossFilePath));
					for (int i = 0; i < lossList.size(); i++) {
						bos.write(((i + 1) + " " + lossList.get(i) + "")
								.getBytes());
						bos.write(System.lineSeparator().getBytes());
					}
					bos.close();

				} catch (IOException e) {
					System.out.println("Exception concering the loss writer ");
				}

				try {

					bos = new BufferedOutputStream(new FileOutputStream(
							WeightFilePath));

					for (int k = 0; k < numofInput + 1; k++) {
						for (int l = 0; l < numofNeuronLayerOne; l++) {
							bos.write((Objects.gtst.getWeights()[k][l] + "")
									.getBytes());
							bos.write(System.lineSeparator().getBytes());
						}
					}

					for (int k = 0; k < numofNeuronLayerOne + 1; k++) {
						for (int l = 0; l < numofNeuronLayerTwo; l++) {
							bos.write((Objects.gtst.getWeights2()[k][l] + "")
									.getBytes());
							bos.write(System.lineSeparator().getBytes());
						}
					}

					for (int k = 0; k < numofNeuronLayerTwo + 1; k++) { // make
																		// to
																		// result
																		// weights
						for (int l = 0; l < 1; l++) {
							bos.write((Objects.gtst.getResultWeights()[k][l] + "")
									.getBytes());
							bos.write(System.lineSeparator().getBytes());

						}
					}

					bos.close();
					System.out.println("Weights saved successfully");
				} catch (IOException e) {
					System.out.println("Exception concering the file writer ");
				}
			}

		} else {
			System.out
					.println("Make sure input is in file arranged with numofSets, numofInput, numofLayerOne ");
			System.out
					.println("numofLayerTwo and then input. Weights saved to file from earlier function \n");

			Objects.pre.runPrediction();

		}// end of else

	}
}
