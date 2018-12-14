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
	String filePath = System.getProperty("user.home") + "\\Desktop\\Models\\";
	String testFilePath =  System.getProperty("user.home") + "\\Desktop\\";
	Layer layer = new Layer(0);
	int numofLayers; 
	int[] layerSizes;  
	double[] mean; 
	double[] strdDev; 
	String[] activationStrings; 
	double[][] weightArray; 
	double[][] inputLayer; 
	double[][] layerValue; 
	
	public void predict(String model, double ...inputs) {
		inputLayer = new double[1][inputs.length];

		acquireModelValues(model); 
		
		for(int i=0; i<inputs.length; i++) {
			inputLayer[0][i] = inputs[i];  
		}
		
		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);
		
		forwardPropagation(); 
		
		listOfValues.clear();
		
		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue)+ "\n");
		
	} //end of predict
	
	public void predict(String modelFilePath, String testFilePath) {
		acquireModelValues(modelFilePath);
		acquireTestValues(testFilePath); 
		
		inputLayer = normalizer.normalizeInputs(inputLayer, mean, strdDev);
		
		forwardPropagation(); 
		
		listOfValues.clear();
		
		System.out.println("Prediction " + java.util.Arrays.deepToString(layerValue) + "\n");
	}
	
	private void formWeightsToArrays(int[] layerSizes) {
		int counter = 0; 
		for(int k=0; k<layerSizes.length-1; k++) {
			weightArray = new double[layerSizes[k]+1][layerSizes[k+1]];
			for(int i=0; i < layerSizes[k]+1; i++) {
				for(int j=0; j < layerSizes[k+1]; j++) {
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

	public void propagateInputLayer() {
		layerValue = appendBiasColumn(inputLayer);
		System.out.println("InitialLayer " + java.util.Arrays.deepToString(layerValue));
		layerValue =  nt.matrixMultiplication(layerValue, weightList.get(0));
		layer.layerValue = layerValue; 
		layer.activation = activationStrings[0];
		layerValue = nt.activate(layer);
	}
	
	public void acquireModelValues(String modelPath) { 
		fr = new FileReader(filePath + modelPath + ".txt");
		fr.initializeFileReader(); //initializes scanner
		fr.readFileIntoList();
		listOfValues = fr.valuesFromFile; 
		numofLayers = (int)(double)listOfValues.get(0); 
		
		listOfValues.remove(0); //remove numofLayers
		
		int numofSets = (int)(double)listOfValues.get(0);
		
		layerSizes = new int[numofLayers];
		mean = new double[numofSets];
		strdDev = new double[numofSets];
		
		
		activationStrings = new String[numofLayers-1];
		
		for(int i=0; i < numofLayers; i++) {
		   layerSizes[i] = (int)(double)listOfValues.get(0);  
		   listOfValues.remove(0);
		}
		for(int i=0; i<numofLayers-1; i++) {
			activationStrings[i] = activator.convertActivationInt((int)(double)listOfValues.get(0));
			listOfValues.remove(0);
		}
		for(int i=0; i<layerSizes[0]; i++) {
			mean[i] = (double)listOfValues.get(0);
			listOfValues.remove(0);
		}
		for(int i=0; i<layerSizes[0]; i++) {
			strdDev[i] = (double)listOfValues.get(0);
			listOfValues.remove(0);
		}
		formWeightsToArrays(layerSizes); 
	}
	
	public void acquireTestValues(String testPath) {
		fr = new FileReader(testFilePath + testPath + ".txt");
		fr.initializeFileReader(); //initializes scanner
		fr.readFileIntoList();
		listOfValues = fr.valuesFromFile; 
		
		int numofSets = (int)listOfValues.size()/(int) layerSizes[0]; 
		
		inputLayer = new double[numofSets][layerSizes[0]];
		
		int counter =0; 
		for(int i=0; i<numofSets; i++) {
			for(int j=0; j<layerSizes[0]; j++) {
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
