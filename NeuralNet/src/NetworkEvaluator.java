import java.util.ArrayList;
import java.util.List;

public class NetworkEvaluator {
	Normalizer normalizer = new Normalizer();
	NetworkTrainer nt = new NetworkTrainer();
	Activator activator = new Activator();
	FileReader fr;
	List<Double> listOfValues = new ArrayList<Double>();
	List<double[][]> weightList = new ArrayList<double[][]>();
	
	List<double[][]> layerList = new ArrayList<double[][]>();
	
	List<Layer> layerListObjects = new ArrayList<Layer>();
	Weights weights = new Weights();
	String filePath = System.getProperty("user.home") + "\\Desktop\\Models\\";
	String testFilePath = System.getProperty("user.home") + "\\Desktop\\";
	Layer layer = new Layer();
	int numofLayers;
	int[] layerSizes;
	int[] layerTypes;
	double[] mean;
	double[] strdDev;
	String[] activationStrings;
	double[][] weightArray;
	double[][] inputLayer;
	double[][] layerValue;

	public void predict(String model, double... inputs) {
		inputLayer = new double[1][inputs.length];

		acquireModelValues(model);

		for (int i = 0; i < inputs.length; i++) {
			inputLayer[0][i] = inputs[i];
		}

		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

	    forwardPropagation();

		listOfValues.clear();

		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue) + "\n");

	} // end of predict

	public void predict(String modelFilePath, String testFilePath) {

		acquireModelValues(modelFilePath);
		acquireTestValues(testFilePath);
		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();

		listOfValues.clear();

		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue) + "\n");
	}

	public void predictNCAA(String modelFilePath, String lookup, String team1, String team2) {

		buildTeamSearch(System.getProperty("user.home") + "\\Desktop\\" + lookup + ".txt");

		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		acquireModelValues(modelFilePath);

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();

		listOfValues.clear();

		System.out.println(
				"Prediction: " + team1 + " vs. " + team2 + " " + java.util.Arrays.deepToString(layerValue) + "\n");
	}

	public void predictNCAALoop(String team1, String team2) {

		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();

		System.out.println("Prediction: " + team1 + " vs. " + team2 + " " + java.util.Arrays.deepToString(layerValue));
		// listOfValues.clear();
	}

	private void predictNCAALoopForBracket(String team1, String team2) {
		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);

		forwardPropagation();
	}

	public void predictNCAAMatch(Normalizer norm, int numofLayers, List<double[][]> weightList, String lookup,
			String team1, String team2) {
		this.numofLayers = numofLayers;

		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		inputLayer = norm.normalizeInputs(inputLayer, norm.meanArray, norm.strdDev);

		forwardPropagationMatch(weightList);

		listOfValues.clear();

		System.out.println(
				"Prediction: " + team1 + " vs. " + team2 + " " + java.util.Arrays.deepToString(layerValue) + "\n");
	}

	public void constructNCAABracket(String model, String starting, String Lookup) {

		ArrayList<Double> valuesFromFile = new ArrayList<Double>();
		ArrayList<String> stringList = new ArrayList<String>();

		String[] top64 = new String[64];
		
		String[] top32 = new String[32];
		String[] top16 = new String[16];
		String[] top8 = new String[8];
		String[] top4 = new String[4];
		String[] championship = new String[2];
		String champion; 

		fr = new FileReader(starting);
		fr.initializeBufferedReader(starting);

		fr.parseDataIntoLists(valuesFromFile, stringList); // values from file == blank

		for(int i=0; i<64; i++) {
			top64[i] = stringList.get(i);
		}
		
		acquireModelValues(model);

		buildTeamSearch(System.getProperty("user.home") + "\\Desktop\\" + Lookup + ".txt");

		for (int i = 0; i < stringList.size(); i += 2) {

			predictNCAALoopForBracket(stringList.get(i), stringList.get(i + 1));

			if (layerValue[0][0] > layerValue[0][1]) {
				top32[i / 2] = stringList.get(i);
			} else {
				top32[i / 2] = stringList.get(i + 1);
			}

		}

		for (int i = 0; i < 32; i += 2) {

			predictNCAALoopForBracket(top32[i], top32[i + 1]);

			if (layerValue[0][0] > layerValue[0][1]) {
				top16[i / 2] = top32[i];
			} else {
				top16[i / 2] = top32[i + 1];
			}

		}

		for (int i = 0; i < 16; i += 2) {

			predictNCAALoopForBracket(top16[i], top16[i + 1]);

			if (layerValue[0][0] > layerValue[0][1]) {
				top8[i / 2] = top16[i];
			} else {
				top8[i / 2] = top16[i + 1];
			}

		}

		for (int i = 0; i < 8; i += 2) {

			predictNCAALoopForBracket(top8[i], top8[i + 1]);

			if (layerValue[0][0] > layerValue[0][1]) {
				top4[i / 2] = top8[i];
			} else {
				top4[i / 2] = top8[i + 1];
			}

		}

		for (int i = 0; i < 4; i += 2) {

			predictNCAALoopForBracket(top4[i], top4[i + 1]);

			if (layerValue[0][0] > layerValue[0][1]) {
				championship[i / 2] = top4[i];
			} else {
				championship[i / 2] = top4[i + 1];
			}

		}
		
		predictNCAALoopForBracket(championship[0], championship[1]);
		
		if(layerValue[0][0] > layerValue[0][1]) {
			champion = championship[0];
		} else {
			champion = championship[1]; 
		}

	//	System.out.println(java.util.Arrays.deepToString()); 
		
		System.out.println(
				"Prediction: " + championship[0] + " vs. " + championship[1] + " " + java.util.Arrays.deepToString(layerValue) + "\n");
		
		formatBracket(top64, top32, top16, top8, top4, championship, champion); 
		 
	}

	private void formatBracket(String[] top64, String[] top32, String[] top16, String[] top8, String[] top4, String[] championship, String champion) {
		int num = findLongestString(top64);
		
		System.out.println(top64[0] + space(10*num - top64[0].length()) + top64[32]);
		System.out.println(top64[1] + space(10*num - top64[1].length()) + top64[33]);
		System.out.println(space(5 + num) + top32[0] + space(10*num - top32[0].length() - 2*(5+num)) + top32[16] + space(5+num));
		System.out.println(top64[2] + space(5+num-top64[2].length()) + top32[1] + space(10*num - top32[1].length() - 2*(5+num)) + top32[17] + space(5+num-top32[17].length()) + top64[34]);
		System.out.println(top64[3] + space(10*num - top64[3].length()) + top64[35]);
		System.out.println(space(2*num) + top16[0] + space(10*num - top16[0].length() - 4*num) + top16[8] + space(2*num));
		System.out.println(space(2*num) + top16[1] + space(10*num - top16[1].length() - 4*num) + top16[9] + space(2*num));
		System.out.println(top64[4] + space(10*num - top64[4].length()) + top64[36]);
		System.out.println(top64[5] + space(5+num-top64[5].length()) + top32[2] + space(10*num - 2*(5+num)  - top32[2].length()) + top32[18] + space(5+num-top32[18].length()) + top64[37]);
		System.out.println(space(5 + num) + top32[3] + space(10*num - top32[3].length() -2*(5+num)) + top32[19] + space(10));
		System.out.println(top64[6] + space(10*num - top64[6].length()) + top64[38]);
		System.out.println(top64[7] + space(10*num - top64[7].length()) + top64[39]);
		System.out.println(space(3*num) + top8[0] + space(10*num - 6*num - top8[0].length()) + top8[4] + space(3*num));
		System.out.println(space(3*num) + top8[1] + space(10*num - 6*num - top8[1].length()) + top8[5] + space(3*num));
		System.out.println(top64[8] + space(10*num - top64[8].length()) + top64[40]);
		System.out.println(top64[9] + space(10*num - top64[9].length()) + top64[41]);
		//16
		System.out.println(space(5 + num) + top32[4] + space(10*num - top32[4].length() -2*(5+num)) + top32[20] + space(5+num));
		System.out.println(top64[10] + space(5+num-top64[10].length()) + top32[5] + space(10*num - top32[5].length() - 2*(5+num)) + top32[21] + space(5+num-top32[21].length()) + top64[42]);
		System.out.println(top64[11] + space(10*num - top64[11].length()) + top64[43]);
		System.out.println(space(2*num) + top16[2] + space(10*num - 4*num - top16[2].length()) + top16[10] + space(30));
		System.out.println(space(2*num) + top16[3] + space(2*num-top16[3].length()) + top4[0] + space(10*num - 8*num - top4[0].length()) + top4[2] + space(2*num-top4[2].length()) + top16[11] + space(2*num));
		
		System.out.println(top64[12] + space(10*num - top64[12].length()) + top64[44]);
		System.out.println(top64[13] + space(5+num-top64[13].length()) + top32[6] + space(10*num - top32[6].length() - 2*(5+num)) + top32[22] + space(5+num-top32[22].length()) + top64[45]);
		System.out.println(space(5 + num) + top32[7] + space(10*num - 2*(5+num) - top32[7].length()) + top32[23] + space(5+num));
		//24	
		System.out.println(top64[14] + space(10*num - top64[14].length()) + top64[46]);
		System.out.println(top64[15] + space(10*num - top64[15].length()) + top64[47]);
		System.out.println(space((int) (4.5*num)) + championship[0] + space(num - championship[0].length()) + championship[1] + space((int) 4.5*num));
		System.out.println(space((int) (5*num)) + champion + space((int)(5*num)));
		//28
		System.out.println(top64[16] + space(10*num - top64[16].length()) + top64[48]);
		System.out.println(top64[17] + space(10*num - top64[17].length()) + top64[49]);
		System.out.println(space(5 + num) + top32[8] + space(10*num - top32[8].length() - 2*(5+num)) + top32[24] + space(10));
		System.out.println(top64[18] + space(5+num-top64[18].length()) + top32[9] + space(10*num - top32[9].length() - 2*(5+num)) + top32[25] + space(5+num-top32[25].length()) + top64[50]);
		System.out.println(top64[19] + space(10*num - top64[19].length()) + top64[51]);
		//33
		System.out.println(space(2*num) + top16[4] + space(2*num - top16[4].length()) + top4[1] + space(10*num - 8*num - top4[1].length() ) + top4[3] + space(2*num-top4[3].length()) + top16[12] + space(2*num));
		System.out.println(space(2*num) + top16[5] + space(10*num - top16[5].length() - 4*num) + top16[13] + space(2*num));
		System.out.println(top64[20] + space(10*num - top64[20].length()) + top64[52]);
		System.out.println(top64[21] + space(5+num-top64[21].length()) + top32[10] + space(10*num - 2*(5+num) - top32[10].length()) + top32[26] + space(5+num-top32[26].length()) + top64[53]);
		System.out.println(space(5 + num) + top32[11] + space(10*num - 2*(5+num) - top32[11].length()) + top32[27] + space(5+num));
		System.out.println(top64[22] + space(10*num - top64[22].length()) + top64[54]);
		System.out.println(top64[23] + space(10*num - top64[23].length()) + top64[55]);
		System.out.println(space(3*num) + top8[2] + space(10*num - top8[2].length() - 6*num) + top8[6] + space(3*num));
		System.out.println(space(3*num) + top8[3] + space(10*num - 6*num - top8[3].length()) + top8[7] + space(3*num));
		System.out.println(top64[24] + space(10*num - top64[24].length()) + top64[56]);
		System.out.println(top64[25] + space(10*num - top64[25].length()) + top64[57]);
		//44
		System.out.println(space(5 + num) + top32[12] + space(10*num - 2*(5+num) - top32[12].length()) + top32[28] + space(5+num));
		System.out.println(top64[26] + space(5+num-top64[26].length()) + top32[13] + space(10*num - 2*(5+num) - top32[13].length()) + top32[29] + space(5+num-top32[29].length()) + top64[58]); 
		System.out.println(top64[27] + space(10*num - top64[27].length()) + top64[59]);
		System.out.println(space(2*num) + top16[6] + space(10*num - 4*num - top16[6].length()) + top16[14] + space(2*num));
		System.out.println(space(2*num) + top16[7] + space(10*num - 4*num - top16[7].length()) + top16[15] + space(2*num));
		System.out.println(top64[28] + space(10*num - top64[28].length()) + top64[60]);
		System.out.println(top64[29] + space(5+num-top64[29].length()) + top32[14] + space(10*num - top32[14].length() - 2*(5+num)) + top32[30] + space(5+num-top32[30].length()) + top64[61]);
		System.out.println(space(5 + num) + top32[15] + space(10*num - top32[15].length() - 2*(5+num)) + top32[31] + space(5+num));
		System.out.println(top64[30] + space(10*num - top64[30].length()) + top64[62]);
		System.out.println(top64[31] + space(10*num - top64[31].length()) + top64[63]);
	}

	private int findLongestString(String[] top64) {
		int n = 0;
		for(int i=0; i<top64.length; i++) {
			if(top64[i].length() > n) {
				n = top64[i].length();
			}
		}
		return n; 
	}
	private String space(int numofSpaces) {
		String space = "";
		for(int i=0; i < numofSpaces; i++) {
			space += " "; 
		}
		return space; 
	}
	
	public void checkModelAgainstActual(String model, String actual, String lookup) {
		int count = 0;

		ArrayList<Double> valuesFromFile = new ArrayList<Double>();
		ArrayList<String> stringList = new ArrayList<String>();

		fr = new FileReader(actual);
		fr.initializeBufferedReader(actual);

		fr.parseDataIntoLists(valuesFromFile, stringList);

		acquireModelValues(model);

		buildTeamSearch(System.getProperty("user.home") + "\\Desktop\\" + lookup + ".txt");

		for (int i = 0; i < stringList.size(); i += 2) {

			predictNCAALoop(stringList.get(i), stringList.get(i + 1));

			if (((layerValue[0][0] > layerValue[0][1]) && (valuesFromFile.get(i) > valuesFromFile.get(i + 1)))
					|| ((layerValue[0][0] < layerValue[0][1]) && (valuesFromFile.get(i) < valuesFromFile.get(i + 1)))) {
				count++;
			}
		}

		System.out.println("Accuracy of model against match " + (count / (double) (stringList.size() / 2)));

	}

	public boolean pairsMatch(Normalizer norm, int numofLayers, List<double[][]> weightList, String matchingTable,
			String lookup) {
		boolean TorF = true;

		ArrayList<Double> valuesFromFile = new ArrayList<Double>();
		ArrayList<String> stringList = new ArrayList<String>();

		fr = new FileReader(matchingTable);
		fr.initializeBufferedReader(matchingTable);

		fr.parseDataIntoLists(valuesFromFile, stringList);

		buildTeamSearch(System.getProperty("user.home") + "\\Desktop\\" + lookup + ".txt");

		for (int i = 0; i < stringList.size(); i += 2) {
			predictNCAAMatch(norm, numofLayers, weightList, lookup, stringList.get(i), stringList.get(i + 1));

			if (((layerValue[0][0] > layerValue[0][1]) && (valuesFromFile.get(i) < valuesFromFile.get(i + 1)))
					|| ((layerValue[0][0] < layerValue[0][1]) && (valuesFromFile.get(i) > valuesFromFile.get(i + 1)))) {
				TorF = false;
				break;
			}
		}

		return TorF;
	}

	private void buildTeamSearch(String lookup) {

		fr = new FileReader(lookup);

		fr.valuesFromFile2 = new ArrayList<Double>();
		fr.stringList2 = new ArrayList<String>();

		fr.initializeBufferedReader();
		fr.parseDataIntoLists(fr.valuesFromFile2, fr.stringList2);
		fr.buildLookupTable();
	}

	private double[] joinArray(double[]... arrays) {
		int length = 0;
		for (double[] array : arrays) {
			length += array.length;
		}

		final double[] result = new double[length];

		int offset = 0;
		for (double[] array : arrays) {
			System.arraycopy(array, 0, result, offset, array.length);
			offset += array.length;
		}

		return result;
	}

	

	private void formWeightsToArrays(int[] layerSizes) {
		int counter = 0;
		for (int k = 0; k < layerSizes.length - 1; k++) {
			weightArray = new double[layerSizes[k] + 1][layerSizes[k + 1]];
			for (int i = 0; i < layerSizes[k] + 1; i++) {
				for (int j = 0; j < layerSizes[k + 1]; j++) {
					weightArray[i][j] = listOfValues.get(counter);
					counter++;
				}
			}
			weightList.add(weightArray);
		}
	}

	public void forwardPropagation() {
		propagateInputLayer();
		for (int i = 1; i < numofLayers - 1; i++) {
			layerValue = appendBiasColumn(layerValue);
			layerValue = nt.matrixMultiplication(layerValue, weightList.get(i));
			layer.layerValue = layerValue;
			layer.activation = activationStrings[i];
			layerValue = nt.activate(layer);
		}
	}

	public void forwardPropagationMatch(List<double[][]> weightList) {
		propagateInputLayerMatch(weightList);
		for (int i = 1; i < numofLayers - 1; i++) {
			layerValue = appendBiasColumn(layerValue);
			layerValue = nt.matrixMultiplication(layerValue, weightList.get(i));
			layer.layerValue = layerValue;
			layer.activation = nt.activatorStrings[i];
			layerValue = nt.activate(layer);
		}
	}

	

	public void propagateInputLayer() {
		layerValue = appendBiasColumn(inputLayer);
		layerValue = nt.matrixMultiplication(layerValue, weightList.get(0));
		layer.layerValue = layerValue;
		layer.activation = activationStrings[0];
		layerValue = nt.activate(layer);
	}

	public void propagateInputLayerMatch(List<double[][]> weightList) {
		layerValue = appendBiasColumn(inputLayer);
		// System.out.println("InitialLayer " +
		// java.util.Arrays.deepToString(layerValue));
		layerValue = nt.matrixMultiplication(layerValue, weightList.get(0));
		layer.layerValue = layerValue;
		layer.activation = nt.activatorStrings[0];
		layerValue = nt.activate(layer);
	}

	public void acquireModelValues(String modelPath) {
		fr = new FileReader(filePath + modelPath + ".txt");

		fr.initializeBufferedReader();
		fr.readDataIntoList();

		listOfValues = fr.valuesFromFile;
		numofLayers = (int) (double) listOfValues.get(0);

		listOfValues.remove(0);

		for (int i = 0; i < numofLayers; i++) {
			listOfValues.remove(0); // used to remove object type since not needed here
		}

		int numofSets = (int) (double) listOfValues.get(0);

		layerSizes = new int[numofLayers];
		mean = new double[numofSets];
		strdDev = new double[numofSets];

		activationStrings = new String[numofLayers - 1];

		for (int i = 0; i < numofLayers; i++) {
			layerSizes[i] = (int) (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		for (int i = 0; i < numofLayers - 1; i++) {
			activationStrings[i] = activator.convertActivationInt((int) (double) listOfValues.get(0));
			listOfValues.remove(0);
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			mean[i] = (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			strdDev[i] = (double) listOfValues.get(0);
			listOfValues.remove(0);
		}
		formWeightsToArrays(layerSizes);
	}

	public void acquireTestValues(String testPath) {
		fr = new FileReader(testFilePath + testPath + ".txt");
		fr.initializeFileReader(); // initializes scanner
		fr.readDoublesFromFileIntoList();
		listOfValues = fr.valuesFromFile;

		int numofSets = (int) listOfValues.size() / (int) layerSizes[0];

		inputLayer = new double[numofSets][layerSizes[0]];

		int counter = 0;
		for (int i = 0; i < numofSets; i++) {
			for (int j = 0; j < layerSizes[0]; j++) {
				inputLayer[i][j] = listOfValues.get(counter);
				counter++;
			}
		}
	}


	public double[][] appendBiasColumn(double[][] layer) {
		double[][] inputsWithBiases = new double[layer.length][layer[0].length + 1];

		for (int i = 0; i < layer.length; i++) {
			for (int j = 0; j < layer[0].length; j++) {
				inputsWithBiases[i][j] = layer[i][j];
			}
		}
		for (int i = 0; i < layer.length; i++) {
			inputsWithBiases[i][layer[0].length] = 1;
		}
		return inputsWithBiases;
	}

}
