import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;

public class Predict {

	static double weightBias = .1;

	public void runPrediction() {

		ArrayList<Double> list = new ArrayList<Double>();
		ArrayList<Double> Inputs = new ArrayList<Double>();
		BufferedReader br = null;
		BufferedReader br2 = null;
		

		
		String weight;
		double weightNum;
		String input;
		double inputNum;
		int numofSets = 0;
		int numofInputs = 0;
		int numofLayerOne = 0;
		int numofLayerTwo = 0;
		double targetBoolean = 0.0; 
		double max = 0.0, min = 0.0, midRange = 0.0 , range = 0.0; 
		double[][] biasedInputs = null;

		double[][] InputArray = null;

		int counter = -1;

		try {
			br = new BufferedReader(new FileReader(
					"C:\\Users\\Miles\\Desktop\\Inputs.txt"));
		
			 
			String nS = br.readLine();
			String nI = br.readLine();
			String nLO = br.readLine();
			String nLT = br.readLine();

			numofSets = Integer.parseInt(nS);
			numofInputs = Integer.parseInt(nI);
			numofLayerOne = Integer.parseInt(nLO);
			numofLayerTwo = Integer.parseInt(nLT);

			InputArray = new double[numofSets][numofInputs];

			while ((input = br.readLine()) != null) {
				inputNum = Double.parseDouble(input);
				Inputs.add(inputNum);
			}

			for (int i = 0; i < numofSets; i++) {
				for (int j = 0; j < numofInputs; j++) {
					counter++;
					InputArray[i][j] = Inputs.get(counter);

				}
			}

			br.close();
		} catch (Exception e) {
			System.out.println("Error occured in inputs " + e);
		}

		double[][] WeightArray;
		double[][] WeightArray2;
		double[][] WeightArrayResult;
		
		

		WeightArray = new double[numofInputs + 1][numofLayerOne];

		WeightArray2 = new double[numofLayerOne + 1][numofLayerTwo];
		WeightArrayResult = new double[numofLayerTwo + 1][1];

		try {

			br2 = new BufferedReader(new FileReader(
					"C:\\Users\\Miles\\Desktop\\Weights.txt"));
			while ((weight = br2.readLine()) != null) { //becuase of the 4 extraDataPoints 
				weightNum = Double.parseDouble(weight);
				list.add(weightNum);
			}
			int weightcounter = -1;

			for (int i = 0; i < numofInputs + 1; i++) {
				for (int j = 0; j < numofLayerOne; j++) {
					weightcounter++;
					WeightArray[i][j] = list.get(weightcounter);
				}
			}

			for (int m = 0; m < numofLayerOne + 1; m++) {
				for (int n = 0; n < numofLayerTwo; n++) {
					weightcounter++;
					WeightArray2[m][n] = list.get(weightcounter);
				}
			}
			for (int o = 0; o < numofLayerTwo + 1; o++) {
				for (int p = 0; p < 1; p++) {
					weightcounter++;
					WeightArrayResult[o][p] = list.get(weightcounter);
				}
			}
			
			 targetBoolean = list.get(weightcounter+3); 
			 int x  = weightcounter+3; 
			
			if(targetBoolean == 0.0) {
				weightcounter++; 
				min = list.get(weightcounter); 
				weightcounter++;
				max = list.get(weightcounter); 
			}
			else if(targetBoolean == 1.0) {
				weightcounter++; 
				midRange = list.get(weightcounter); 
				weightcounter++;
				range = list.get(weightcounter); 
			} 
			
		
			
			
			double[] mean = new double[numofInputs];
			double[] strdDev = new double[numofInputs]; 
			
			for(int i=0; i<numofInputs; i++) {
				x++; 
				mean[i] = list.get(x);
			}
			for(int j=0; j<numofInputs; j++ ){
				x++;
				strdDev[j] = list.get(x);  
			}
			
		
			InputArray = Objects.gtst.normalize(InputArray, numofSets, numofInputs, mean, strdDev, true); 
		//	System.out.println(java.util.Arrays.deepToString(InputArray));
			
			//FIX THIS SO THE INPUTS ARE NORMALIZED ACCORDING TO WHEN THE WEIGHTS WERE SAVED

			biasedInputs = new double[numofSets][numofInputs + 1];
			for (int j = 0; j < numofSets; j++) {
				for (int i = 0; i < numofInputs; i++) {
					biasedInputs[j][i] = InputArray[j][i];
				}
			} // gets inputs, STILL without biases

			for (int i = 0; i < numofSets; i++) {
				biasedInputs[i][numofInputs] = 1; // sets biases to one
			}

			Objects.gtst.setInputs(biasedInputs);
			/*
			 * for (int i = 0; i < numofLayerOne; i++) { // wrong
			 * WeightArray[numofInputs][i] = weightBias; } // adds bias to
			 * weights
			 */
			// WeightArray = Objects.gtst.addWeightBiases(WeightArray);
			// WeightArray2 = Objects.gtst.addWeightBiases(WeightArray2);
			// WeightArrayResult =
			// Objects.gtst.addWeightBiases(WeightArrayResult);

			Objects.gtst.setWeights(WeightArray);

			// for (int i = 0; i < numofLayerTwo; i++) {
			// WeightArray2[numofLayerOne][i] = weightBias;
			// }
			Objects.gtst.setWeights2(WeightArray2);

			// WeightArrayResult[numofLayerTwo][0] = weightBias;

			Objects.gtst.setResultWeights(WeightArrayResult);
/*
			System.out.println("1 "
					+ java.util.Arrays.deepToString(Objects.gtst.getWeights()));
			System.out
					.println("2 "
							+ java.util.Arrays.deepToString(Objects.gtst
									.getWeights2()));
			System.out.println("3 "
					+ java.util.Arrays.deepToString(Objects.gtst
							.getResultWeights()));
*/
			br2.close();

		} catch (Exception e) {
			System.out.println("Possible error in weights");
		}

		Objects.fdp.CreateLayer(Objects.gtst.getInputs());
		Objects.fdp.CreateSecondLayer(Objects.gtst.getLayerOne());
		Objects.fdp.CreateResult(Objects.gtst.getLayerTwo());
		
	

		
		if(targetBoolean == 0.0) {
			System.out.println("Prediction pre-InverseMinMax " + java.util.Arrays.deepToString(Objects.gtst.getResult())); 
		System.out.println("Prediction "
				+ java.util.Arrays.deepToString(Objects.gtst.InverseMinMax(Objects.gtst.getResult(), max, min)));
		} 
		else if(targetBoolean == 1.0) {
			System.out.println("Prediction "
					+ java.util.Arrays.deepToString(Objects.gtst.InverseRangeNormalize(Objects.gtst.getResult(), range, midRange)));
			
		} else { 
			System.out.println("Prediction "
					+ java.util.Arrays.deepToString(Objects.gtst.getResult()));
		}
		
	}
}
