import java.util.ArrayList;
import java.util.List;

/**
 * Class network evaluator is used to take existing network models and predict information 
 * based on those models
 */
public class NetworkEvaluator {
	Normalizer normalizer = new Normalizer();
	NetworkTrainer nt = new NetworkTrainer();
	Activator activator = new Activator();
	FileReader fr;

	List<Double> listOfValues = new ArrayList<Double>();
	Weights weights = new Weights();
	ForwardPropagator fp = new ForwardPropagator();

	String strdFilePath = System.getProperty("user.home") + "\\Desktop\\NeuralNetworkRelated\\";

	int numofLayers;
	int[] layerSizes;
	double[] mean;
	double[] strdDev;
	String[] activationStrings;
	double[][] inputLayer;
	double[][] finalLayer;

	
	/**
	 * The function predict takes a model from the models folder and reads in information from the 
	 * arguments list and predicts the output layer based on that input
	 * 
	 * Acquires the values from the model "model" 
	 * Then reads the inputs from the argument list into a two dimensional array so it can be used with the 
	 * forward propagation functions. 
	 * 
	 * The inputs are normalized according to the model
	 * 
	 * The model is created and then the objects for forward propagation are created 
	 * 
	 * Finally, forward propagation is run and the output displayed
	 * 
	 * 
	 * @param model: String model is the name of the network model used for evaluation 
	 * @param inputs: a variable amount of inputs used in evaluation 
	 */
	public void predict(String model, double... inputs) {
		acquireModelValues(model);

		inputLayer = new double[1][inputs.length];

		for (int i = 0; i < inputs.length; i++) {
			inputLayer[0][i] = inputs[i];
		}

		normalizer.normalizeInputs(inputLayer, mean, strdDev);

		buildNetworkModel();

		fp.constructForwardPropagationObjects(nm, weights);

		fp.runPropagation(); // at the moment won't work properly with dropout layer

		System.out.println("Prediction "
				+ java.util.Arrays.deepToString(fp.layerList.get(fp.layerList.size() - 1).layerValue) + "\n");
	}

	/**
	 * The function predict takes a model from models folder and reads information from 
	 * testFilePath to evaluate with.  
	 *  
	 * Acquires model values from the model "model"
	 * Acquires input values from the file "testFilePath" 
	 *  
	 * The inputs are normalized according to the model
	 * 
	 * The model is created and then the objects for forward propagation are created 
	 * 
	 * Finally, forward propagation is run and the output displayed
	 *  
	 * @param modelFilePath: String model is the name of the network model used for evaluation 
	 * @param testFilePath: String testFilePath is the name of the file containing the input values 
	 */
	public void predict(String modelFilePath, String testFilePath) {
		acquireModelValues(modelFilePath);
		acquireTestValues(testFilePath);

		normalizer.normalizeInputs(inputLayer, mean, strdDev);

		buildNetworkModel();

		fp.constructForwardPropagationObjects(nm, weights);

		fp.runPropagation(); // at the moment won't work properly with dropout layer

		System.out.println("Prediction "
				+ java.util.Arrays.deepToString(fp.layerList.get(fp.layerList.size() - 1).layerValue) + "\n");
	}

	/**
	 * The function predict NCAA match up predicts a game between two teams using a precalculated model and lookup file
	 * 
	 * A table is created that formats the statistics of all the teams in the lookup file
	 * 
	 * The statistics of the first and second team are stored 
	 * 
	 * The to create the game the stats are joined together 
	 * 
	 * The network model is read and formatted into a networkModel object 
	 * 
	 * The input layer is created by reading in the team stats and converting it to a two
	 * dimensional array 
	 * 
	 * The inputs are normalized according to the model
	 * 
	 * The model is created and then the objects for forward propagation are created 
	 * 
	 * Finally, forward propagation is run and the output displayed
	 *
	 * @param modelFilePath: The string modelFilePath is the name of the networkModel used in evaluation 
	 * @param lookup: The lookup file which contains all the stats of the teams 
	 * @param team1: A team to play 
	 * @param team2 The opposing team to play 
	 */
	public void predictNCAAMatchup(String modelFilePath, String lookup, String team1, String team2) {

		buildTeamSearch(strdFilePath + lookup + ".txt");

		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		acquireModelValues(modelFilePath);

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		normalizer.normalizeInputs(inputLayer, mean, strdDev);

		buildNetworkModel();

		fp.constructForwardPropagationObjects(nm, weights);

		fp.runPropagation(); // at the moment won't work properly with dropout layer

		System.out.println("Prediction: " + team1 + " vs. " + team2 + " "
				+ java.util.Arrays.deepToString(fp.layerList.get(fp.layerList.size() - 1).layerValue) + "\n");
	}

	/**
	 * predictNCAALoop is built to predict the outcome of many different mathces at once as it is used in a loop
	 * 
	 * It uses the function predictNCAALoopNoOutput as it is and then displays the output 
	 * 
	 * @param team1: A team to play
	 * @param team2; The opposing team to play 
	 */
	public void predictNCAALoop(String team1, String team2) {

		predictNCAALoopNoOutput(team1, team2);

		System.out.println("Prediction: " + team1 + " vs. " + team2 + " "
				+ java.util.Arrays.deepToString(fp.layerList.get(fp.layerList.size() - 1).layerValue) + "\n");
	}

	
	/**
	 * predictNCAALoopNoOutput is used to predict many game outcomes one after the other 
	 * 
	 * This function is used in conjunction with another function and not on its own as the model and 
	 * team search are assumed to be initialized and run 
	 * 
	 * The first and second teams stats are read in using a text search in the fileReader
	 * 
 	 * The to create the game the stats are joined together 
	 * 
	 * The network model is read and formatted into a networkModel object 
	 * 
	 * The input layer is created by reading in the team stats and converting it to a two
	 * dimensional array 
	 * 
	 * The inputs are normalized according to the model
	 * 
	 * The model is created and then the objects for forward propagation are created 
	 * 
	 * Finally, forward propagation is run and the output displayed
	 *
	 * @param team1: A team to play 
	 * @param team2: An opposing team to play 
	 */
	private void predictNCAALoopNoOutput(String team1, String team2) {
		double[] team1Stats = fr.textSearch(team1);

		double[] team2Stats = fr.textSearch(team2);

		double[] game = joinArray(team1Stats, team2Stats);

		inputLayer = new double[1][game.length];

		for (int i = 0; i < game.length; i++) {
			inputLayer[0][i] = game[i];
		}

		normalizer.normalizeInputs(inputLayer, mean, strdDev);

		buildNetworkModel();

		fp.constructForwardPropagationObjects(nm, weights);

		fp.runPropagation(); // at the moment won't work properly with dropout layer

		finalLayer = fp.layerList.get(fp.layerList.size() - 1).layerValue;
	}

	/**
	 * The function constructNCAABraket is used to calculate an entire NCAA bracket using network model 
	 * Lists are initialized to store strings and numbers separately 
	 * 
	 * Space is allocated for arrays of strings of various levels in the bracket 
	 * 
	 * A fileReader is initialized to read from the starting file which is a file containing the first games 
	 * of the bracket. The rest of the bracket is determined by the neural network 
	 * 
	 * The data is read and filtered into two lists. One of doubles and one of strings 
	 * 
	 * The model and team search are built and then the initial bracket level created from the string list 
	 * 
	 * Subsequent levels of the bracket are evaluated and then the entire bracket is displayed 
	 * 
	 * @param model: The network model used for evaluation 
	 * @param starting: The filePath for the starting 64 teams in the bracket 
	 * @param Lookup: The lookup file containing the stats of each team 
	 */
	public void constructNCAABracket(String model, String starting, String Lookup) {

		ArrayList<Double> valuesFromFile = new ArrayList<Double>();
		ArrayList<String> stringList = new ArrayList<String>();

		String[] top64 = new String[64];
		String[] top32 = new String[32];
		String[] top16 = new String[16];
		String[] top8 = new String[8];
		String[] top4 = new String[4];
		String[] championship = new String[2];
		String[] champion = new String[1];

		fr = new FileReader(starting);
		fr.initializeBufferedReader(starting);

		fr.parseDataIntoLists(valuesFromFile, stringList); // values from file == blank

		acquireModelValues(model);

		buildTeamSearch(strdFilePath + Lookup + ".txt");

		for (int i = 0; i < 64; i++) {
			top64[i] = stringList.get(i);
		}

		evaluateBracketLevel(top32, top64, 64);

		evaluateBracketLevel(top16, top32, 32);

		evaluateBracketLevel(top8, top16, 16);

		evaluateBracketLevel(top4, top8, 8);

		evaluateBracketLevel(championship, top4, 4);

		evaluateBracketLevel(champion, championship, 2);

		System.out.println("Prediction: " + championship[0] + " vs. " + championship[1] + " "
				+ java.util.Arrays.deepToString(finalLayer) + "\n");

		formatBracket(top64, top32, top16, top8, top4, championship, champion);
	}

	/**
	 * function used to evaluate the bracket at different levels
	 * cycles through number of teams in that level (i+=2 because 2 teams are evaluated at a time)
	 * 
	 * predicts game by running the predictNCAA function in the higher level of the bracket
	 * depending on the outcome of this game the winning team moves to the lower level
	 * 
	 * the prediction made by the predictNCAA function is printed added to as a string to the name of the team
	 * 
	 * @param lowerLevel: an empty string array that holds the teams that make it to the next level as determined by the higher level games
	 * @param higherLevel; a string array holding the names of the team that are playing at this stage of the evaluation 
	 * @param numofTeamsInLevel: the number of teams in the higher level 
	 */
	private void evaluateBracketLevel(String[] lowerLevel, String[] higherLevel, int numofTeamsInLevel) {
		for (int i = 0; i < numofTeamsInLevel; i += 2) {

			predictNCAALoopNoOutput(higherLevel[i], higherLevel[i + 1]);

			if (finalLayer[0][0] > finalLayer[0][1]) {
				lowerLevel[i / 2] = higherLevel[i];
			} else {
				lowerLevel[i / 2] = higherLevel[i + 1];
			}

			higherLevel[i] += " " + Math.round(finalLayer[0][0] * 1000) / 1000.0;
			higherLevel[i + 1] += " " + Math.round(finalLayer[0][1] * 1000) / 1000.0;
		}
	}

	/**
	 * formats the bracket using spacing of the largest team name
	 * @param top64: top 64 teams in the bracket
	 * @param top32: top 32 teams in the bracket
	 * @param top16: top 16 teams in the bracket
	 * @param top8: top 8 teams in the bracket
	 * @param top4: the final four teams in the bracket 
	 * @param championship: the teams in the championship game 
	 * @param champion: the winner of the championship game 
	 */
	private void formatBracket(String[] top64, String[] top32, String[] top16, String[] top8, String[] top4,
			String[] championship, String[] champion) {
		int num = findLongestString(top64);

		System.out.println(top64[0] + space(10 * num - top64[0].length()) + top64[32]);
		System.out.println(top64[1] + space(10 * num - top64[1].length()) + top64[33]);
		System.out.println(space(5 + num) + top32[0] + space(10 * num - top32[0].length() - 2 * (5 + num)) + top32[16]
				+ space(5 + num));
		System.out.println(top64[2] + space(5 + num - top64[2].length()) + top32[1]
				+ space(10 * num - top32[1].length() - 2 * (5 + num)) + top32[17] + space(5 + num - top32[17].length())
				+ top64[34]);
		System.out.println(top64[3] + space(10 * num - top64[3].length()) + top64[35]);
		System.out.println(
				space(2 * num) + top16[0] + space(10 * num - top16[0].length() - 4 * num) + top16[8] + space(2 * num));
		System.out.println(
				space(2 * num) + top16[1] + space(10 * num - top16[1].length() - 4 * num) + top16[9] + space(2 * num));

		System.out.println(top64[4] + space(10 * num - top64[4].length()) + top64[36]);
		System.out.println(top64[5] + space(5 + num - top64[5].length()) + top32[2]
				+ space(10 * num - 2 * (5 + num) - top32[2].length()) + top32[18] + space(5 + num - top32[18].length())
				+ top64[37]);
		System.out.println(space(5 + num) + top32[3] + space(10 * num - top32[3].length() - 2 * (5 + num)) + top32[19]
				+ space(10));
		System.out.println(top64[6] + space(10 * num - top64[6].length()) + top64[38]);
		System.out.println(top64[7] + space(10 * num - top64[7].length()) + top64[39]);
		System.out.println(
				space(3 * num) + top8[0] + space(10 * num - 6 * num - top8[0].length()) + top8[4] + space(3 * num));
		System.out.println(
				space(3 * num) + top8[1] + space(10 * num - 6 * num - top8[1].length()) + top8[5] + space(3 * num));

		System.out.println(top64[8] + space(10 * num - top64[8].length()) + top64[40]);
		System.out.println(top64[9] + space(10 * num - top64[9].length()) + top64[41]);
		System.out.println(space(5 + num) + top32[4] + space(10 * num - top32[4].length() - 2 * (5 + num)) + top32[20]
				+ space(5 + num));
		System.out.println(top64[10] + space(5 + num - top64[10].length()) + top32[5]
				+ space(10 * num - top32[5].length() - 2 * (5 + num)) + top32[21] + space(5 + num - top32[21].length())
				+ top64[42]);
		System.out.println(top64[11] + space(10 * num - top64[11].length()) + top64[43]);
		System.out.println(
				space(2 * num) + top16[2] + space(10 * num - 4 * num - top16[2].length()) + top16[10] + space(30));
		System.out.println(space(2 * num) + top16[3] + space(2 * num - top16[3].length()) + top4[0]
				+ space(10 * num - 8 * num - top4[0].length()) + top4[2] + space(2 * num - top4[2].length()) + top16[11]
				+ space(2 * num));

		System.out.println(top64[12] + space(10 * num - top64[12].length()) + top64[44]);
		System.out.println(top64[13] + space(5 + num - top64[13].length()) + top32[6]
				+ space(10 * num - top32[6].length() - 2 * (5 + num)) + top32[22] + space(5 + num - top32[22].length())
				+ top64[45]);
		System.out.println(space(5 + num) + top32[7] + space(10 * num - 2 * (5 + num) - top32[7].length()) + top32[23]
				+ space(5 + num));
		System.out.println(top64[14] + space(10 * num - top64[14].length()) + top64[46]);
		System.out.println(top64[15] + space(10 * num - top64[15].length()) + top64[47]);
		System.out.println(space((int) (4.5 * num)) + championship[0] + space(num - championship[0].length())
				+ championship[1] + space((int) 4.5 * num));
		System.out.println(space((int) (5 * num)) + "  " + champion[0] + space((int) (5 * num))); // two spaces added
																									// for the decimal

		System.out.println(top64[16] + space(10 * num - top64[16].length()) + top64[48]);
		System.out.println(top64[17] + space(10 * num - top64[17].length()) + top64[49]);
		System.out.println(space(5 + num) + top32[8] + space(10 * num - top32[8].length() - 2 * (5 + num)) + top32[24]
				+ space(10));
		System.out.println(top64[18] + space(5 + num - top64[18].length()) + top32[9]
				+ space(10 * num - top32[9].length() - 2 * (5 + num)) + top32[25] + space(5 + num - top32[25].length())
				+ top64[50]);
		System.out.println(top64[19] + space(10 * num - top64[19].length()) + top64[51]);
		System.out.println(space(2 * num) + top16[4] + space(2 * num - top16[4].length()) + top4[1]
				+ space(10 * num - 8 * num - top4[1].length()) + top4[3] + space(2 * num - top4[3].length()) + top16[12]
				+ space(2 * num));
		System.out.println(
				space(2 * num) + top16[5] + space(10 * num - top16[5].length() - 4 * num) + top16[13] + space(2 * num));

		System.out.println(top64[20] + space(10 * num - top64[20].length()) + top64[52]);
		System.out.println(top64[21] + space(5 + num - top64[21].length()) + top32[10]
				+ space(10 * num - 2 * (5 + num) - top32[10].length()) + top32[26] + space(5 + num - top32[26].length())
				+ top64[53]);
		System.out.println(space(5 + num) + top32[11] + space(10 * num - 2 * (5 + num) - top32[11].length()) + top32[27]
				+ space(5 + num));
		System.out.println(top64[22] + space(10 * num - top64[22].length()) + top64[54]);
		System.out.println(top64[23] + space(10 * num - top64[23].length()) + top64[55]);
		System.out.println(
				space(3 * num) + top8[2] + space(10 * num - top8[2].length() - 6 * num) + top8[6] + space(3 * num));
		System.out.println(
				space(3 * num) + top8[3] + space(10 * num - 6 * num - top8[3].length()) + top8[7] + space(3 * num));

		System.out.println(top64[24] + space(10 * num - top64[24].length()) + top64[56]);
		System.out.println(top64[25] + space(10 * num - top64[25].length()) + top64[57]);
		System.out.println(space(5 + num) + top32[12] + space(10 * num - 2 * (5 + num) - top32[12].length()) + top32[28]
				+ space(5 + num));
		System.out.println(top64[26] + space(5 + num - top64[26].length()) + top32[13]
				+ space(10 * num - 2 * (5 + num) - top32[13].length()) + top32[29] + space(5 + num - top32[29].length())
				+ top64[58]);
		System.out.println(top64[27] + space(10 * num - top64[27].length()) + top64[59]);
		System.out.println(
				space(2 * num) + top16[6] + space(10 * num - 4 * num - top16[6].length()) + top16[14] + space(2 * num));
		System.out.println(
				space(2 * num) + top16[7] + space(10 * num - 4 * num - top16[7].length()) + top16[15] + space(2 * num));

		System.out.println(top64[28] + space(10 * num - top64[28].length()) + top64[60]);
		System.out.println(top64[29] + space(5 + num - top64[29].length()) + top32[14]
				+ space(10 * num - top32[14].length() - 2 * (5 + num)) + top32[30] + space(5 + num - top32[30].length())
				+ top64[61]);
		System.out.println(space(5 + num) + top32[15] + space(10 * num - top32[15].length() - 2 * (5 + num)) + top32[31]
				+ space(5 + num));
		System.out.println(top64[30] + space(10 * num - top64[30].length()) + top64[62]);
		System.out.println(top64[31] + space(10 * num - top64[31].length()) + top64[63]);
	}

	/**
	 * Returns the size of the largest string in a string array 
	 * @param top64: the top64 teams in the bracket as they are used for bracket formatting 
	 * @return: returns the length of the longest string in the array 
	 */
	private int findLongestString(String[] top64) {
		int n = 0;
		for (int i = 0; i < top64.length; i++) {
			if (top64[i].length() > n) {
				n = top64[i].length();
			}
		}
		return n;
	}
	
	/**
	 * returns the a string with the number of spaces specified in the arguments 
	 * @param numofSpaces: the number of whitespace characters to be added to the string 
	 * @return: returns a string with the specified number of whitespace characters
	 */
	private String space(int numofSpaces) {
		String space = "";
		for (int i = 0; i < numofSpaces; i++) {
			space += " ";
		}
		return space;
	}

	/**
	 * checks the accuracy of a model from the file system with the actual results which are stored as a file in the file system
	 * 
	 * reads the information into the lists separating them by name (stringList) and by double value
	 * then reads the model value using the acquireModelValues function 
	 * builds a table to search teams with (the team and their corresponding statistics) 
	 * predicts a match using the predictNCAALoopNoOutput function 
	 * the prediction is compared to the actual result from the actual file
	 * the accuracy is then computed and displayed 
	 * 
	 * @param model: the model to check the accuracy of 
	 * @param actual: the actual results of the matches 
	 * @param lookup: the name of the lookup file which holds the information for each team 
	 */
	public void checkModelAgainstActual(String model, String actual, String lookup) {
		int count = 0;

		ArrayList<Double> valuesFromFile = new ArrayList<Double>(); 
		ArrayList<String> stringList = new ArrayList<String>();

		fr = new FileReader(actual);
		fr.initializeBufferedReader(actual);
		fr.parseDataIntoLists(valuesFromFile, stringList);

		acquireModelValues(model);

		buildTeamSearch(strdFilePath + lookup + ".txt");

		for (int i = 0; i < stringList.size(); i += 2) {

			predictNCAALoopNoOutput(stringList.get(i), stringList.get(i + 1));

			if (((finalLayer[0][0] > finalLayer[0][1]) && (valuesFromFile.get(i) > valuesFromFile.get(i + 1)))
					|| ((finalLayer[0][0] < finalLayer[0][1]) && (valuesFromFile.get(i) < valuesFromFile.get(i + 1)))) {
				count++;
			}
		}

		System.out.println("Accuracy of model against match " + (count / (double) (stringList.size() / 2)));
	}

	/**
	 * 
	 * @param lookup
	 */
	private void buildTeamSearch(String lookup) {
		fr = new FileReader(lookup);

		fr.valuesFromFile2 = new ArrayList<Double>();
		fr.stringList2 = new ArrayList<String>();

		fr.initializeBufferedReader();
		fr.parseDataIntoLists(fr.valuesFromFile2, fr.stringList2);
		fr.buildLookupTable();
	}

	/**
	 * combines the values of arbitrarily many single dimensional double arrays by concatenation
	 * 
	 * determines the total size of the array by adding the sizes of all the arrays entered 
	 * concatenates by using an enhanced for loop to copy the array into the larger result and 
	 * then add an offset to tell the system where to start copying the next array 
	 * 
	 * @param arrays: the arrays to be combined sequentially 
	 * @return returns the fully combined array 
	 */
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

	double[][] weightArray;

	/**
	 * takes a list of weight values and then forms them into the properly sized arrays as a list of two dimensional arrays
	 * cycles through the number of weight layers that need to be created 
	 * allocates space for the array
	 * loops through the list adding weights to the array from left to right
	 * 
	 * adds the array to the weightList
	 * 
	 * @param layerSizes: an array of layer sizes that the weights need to accommodate
	 */
	private void formWeightsToArrays(int[] layerSizes) {

		for (int k = 0; k < layerSizes.length - 1; k++) { //cycle through as many times are there are layers minus one (the last layer doesn't need weights)
			weightArray = new double[layerSizes[k] + 1][layerSizes[k + 1]]; //first dimension equals the first layer size plus one for the bias
																			// second dimension equals the size of the second layer
			for (int i = 0; i < layerSizes[k] + 1; i++) {
				for (int j = 0; j < layerSizes[k + 1]; j++) {
					weightArray[i][j] = listOfValues.get(getCount());
				}
			}
			weights.weightList.add(weightArray);
		}
	}

	int numofInput;
	NetworkModel nm;

	/**
	 * builds a network model to use for evaluation 
	 * creates a network model object
	 * adds an input layer to that model
	 * adds the hidden layers to the model based on the size of the model 
	 * adds an output layer to the model
	 */
	public void buildNetworkModel() {
		nm = new NetworkModel();

		nm.buildInputLayerEvaluator(numofInput, inputLayer); //the number of columns, an inputLayer object

		for (int i = 0; i < numofLayers - 2; i++) {
			nm.buildHiddenLayer(layerSizes[i + 1], activationStrings[i]);
		}

		nm.buildOutputLayer(layerSizes[layerSizes.length - 1], activationStrings[activationStrings.length - 1]);
	}

	String modelFilePath = System.getProperty("user.home") + "\\Desktop\\Models\\";

	/**
	 * method acquireModelValues opens a model file from the file system and reads it into a usable
	 * model that the evaluator can use
	 * 
	 * initializes a file reader and reads the values into a list
	 * The model files are always formatted in the same way (as they are created by the same program) 
	 * The number of layers in the model is the first value in the list and is assigned
	 * Each time something is accessed from the list a function getCount is called that returns 
	 * the current location to be read from the list and then increments the count 
	 * 
	 * @param modelPath: the string name of the file 
	 */
	public void acquireModelValues(String modelPath) {
		fr = new FileReader(modelFilePath + modelPath + ".txt");

		fr.initializeBufferedReader();
		fr.readDataIntoList();

		listOfValues = fr.valuesFromFile;

		numofLayers = (int) (double) listOfValues.get(getCount());

		for (int i = 0; i < numofLayers; i++) {
			getCount();
		}

		layerSizes = new int[numofLayers];
		activationStrings = new String[numofLayers - 1];

		for (int i = 0; i < numofLayers; i++) {
			layerSizes[i] = (int) (double) listOfValues.get(getCount());
		}

		numofInput = layerSizes[0];
		mean = new double[numofInput];
		strdDev = new double[numofInput];

		for (int i = 0; i < numofLayers - 1; i++) {
			activationStrings[i] = activator.convertActivationInt((int) (double) listOfValues.get(getCount()));
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			mean[i] = (double) listOfValues.get(getCount());
		}
		for (int i = 0; i < layerSizes[0]; i++) {
			strdDev[i] = (double) listOfValues.get(getCount());
		}
		formWeightsToArrays(layerSizes);
	}

	int count = 0;

	/**
	 * function getCount returns the value of the current location to be read in the values list
	 * increments the count each time the function is called
	 * @return
	 */
	private int getCount() {
		return count++; 
	}

	/**
	 * method acquireTestValues reads the input file into a format that can be used 
	 * for network evaluation. These are the inputs the are operated on to produce the final result 
	 * 
	 * numerical values are read in from the file with the file reader 
	 * 
	 * the number of sets are calculated using the number of entries in the list and the layerSizes read from the model
	 * 
	 * The values are then read into a properly sized array
	 * @param testPath
	 */
	public void acquireTestValues(String testPath) {
		fr = new FileReader(strdFilePath + testPath + ".txt");
		fr.initializeFileReader(); // initializes scanner
		fr.readDoublesFromFileIntoList();
		listOfValues = fr.valuesFromFile;

		int numofSets = (int) listOfValues.size() / (int) layerSizes[0]; //layer sizes are calculated when the model is read 

		inputLayer = new double[numofSets][layerSizes[0]];

		int counter = 0;
		for (int i = 0; i < numofSets; i++) {
			for (int j = 0; j < layerSizes[0]; j++) {
				inputLayer[i][j] = listOfValues.get(counter);
				counter++;
			}
		}
	}

}
