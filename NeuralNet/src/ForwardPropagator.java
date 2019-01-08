import java.util.ArrayList;
import java.util.List;

public class ForwardPropagator {
	int objectTracker = 0;
	static int layerCounter = 0; 
	int testObjectTracker = 0;
	static int batchSize = 1; 
	static int remainingBatchSize = 0; 
	double[][] layerValue;
	double[][] currentBatch; 
	ForwardPropagator forwardPropObj;
	List<ForwardPropagator> propagationObjects = new ArrayList<ForwardPropagator>();
	List<ForwardPropagator> testPropagationObjects = new ArrayList<ForwardPropagator>();
	static List<Layer> layerList = new ArrayList<Layer>(); 
	static List<double[][]> weightList = new ArrayList<double[][]>();  
	static InputLayer inputLayer;
	static NetworkTrainer nt = new NetworkTrainer(); 
	Activator activator = new Activator(); 
	
	
	public double[][] propagate(Layer layer, Layer nextLayer) { 
		
		layerValue = propagationObjects.get(objectTracker).propagate(layer, nextLayer);
	
		if(objectTracker == (propagationObjects.size()-1)) {
			objectTracker = 0;
			layerCounter  = 0; 
		} else {
			objectTracker++;
			layerCounter++; 
		}
	
	
		return layerValue;
	}
	
	public double[][] propagateTest(Layer layer, Layer nextLayer) { 
		if(testObjectTracker == 0) {
			layerCounter = 0;
		}
		
		layerValue = testPropagationObjects.get(testObjectTracker).propagate(layer, nextLayer);
	
		if(testObjectTracker == (testPropagationObjects.size()-1)) {
			testObjectTracker = 0;
			layerCounter  = 0; 
		} else {
			testObjectTracker++;
			layerCounter++; 
		}
	
		
		return layerValue;
	}

	public void constructForwardPropagationObjects(List<Layer> layerList, List<double[][]> weightList) { //only occur once
		
		setupConstants(layerList, weightList); 
		if(inputLayer.numofSets > 140) {
			forwardPropObj = new TestPropagator();
			testPropagationObjects.add(forwardPropObj);
		}
		
		for (int i = 0; i < layerList.size()-1; i++) { // minus one because returns last value
			if (layerList.get(i) instanceof HiddenLayer) {
				forwardPropObj = new DensePropagator();
			} else if (layerList.get(i) instanceof InputLayer) {
				forwardPropObj = new InputLayerPropagator();
			} else if (layerList.get(i) instanceof ConvolutionalLayer) {
				forwardPropObj = new ConvolutionalPropagator();
			} else if (layerList.get(i) instanceof PoolingLayer) {
				forwardPropObj = new PoolingPropagator();
			}

			propagationObjects.add(forwardPropObj);
			
			if(inputLayer.numofSets > 140) {
				if(i>0) {
					testPropagationObjects.add(forwardPropObj);
				}
			}
		}
	}
	
	private void setupConstants(List<Layer> layerList, List<double[][]> weightList) {
		ForwardPropagator.layerList = layerList; 		
		inputLayer = (InputLayer) layerList.get(0); 
		batchSize = inputLayer.batchSize; 
		remainingBatchSize = inputLayer.remainingBatchSize;
		ForwardPropagator.weightList = weightList; 
	}
	public double[][] appendBiasColumn(Layer layer) {
		double[][] layerValue = copyArray(layer.layerValue);
		double[][] inputsWithBiases = new double[layerValue.length][layerValue[0].length + 1];

		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				inputsWithBiases[i][j] = layerValue[i][j];
			}
		}
		for (int i = 0; i < layerValue.length; i++) {
			inputsWithBiases[i][layerValue[0].length] = 1;
		}
		return inputsWithBiases;
	}

	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer);
		return activatedValue;
	}
	
	public double[][] copyArray(double[][] input) {
		double[][] copy = new double[input.length][input[0].length];
		for (int i = 0; i < input.length; i++) {
			for (int j = 0; j < input[0].length; j++) {
				copy[i][j] = input[i][j];
			}
		}
		return copy;
	}

	
}

class InputLayerPropagator extends ForwardPropagator {
	
	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] layerValue;
		currentBatch = getBatch(layer); 
		layer.currentBatch = currentBatch; 
		layerValue = nt.matrixMultiplication(currentBatch, weightList.get(0));
		nextLayer.preActivatedValue = layerValue;
		nextLayer.layerValue = layerValue; //it's here for a reason
		nextLayer.layerValue = activate(nextLayer);
		return nextLayer.layerValue; 
	}
	
	int batchTracker = 0;
	int batchCounter = 0;

	public double[][] getBatch(Layer layer) {
		double[][] batch;

		if (remainingBatchSize == 0) {
			remainingBatchSize = batchSize;
		}

		if (!hasReachedEndofBatch()) {
			batch = new double[batchSize][layer.layerSize + 1];
			for (int i = 0; i < batchSize; i++) {
				for (int j = 0; j < layer.layerSize + 1; j++) { // added one for the bias column
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++;
			}
			batchCounter++;

		} else {
			batch = new double[remainingBatchSize][layer.layerSize + 1];
			for (int i = 0; i < remainingBatchSize; i++) {
				for (int j = 0; j < layer.layerSize + 1; j++) {
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++;
			}
			batchTracker = 0;
			batchCounter = 0;
		}
		return batch;
	}

	public boolean hasReachedEndofBatch() {
		if (nt.numofBatches - 1 == batchCounter) {
			return true;
		} else {
			return false;
		}
	}
	
	
}

class DensePropagator extends ForwardPropagator {
	
	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] layerValue; 
		layer.layerValue = appendBiasColumn(layer);
		layerValue = nt.matrixMultiplication(layer.layerValue, weightList.get(layerCounter));
		nextLayer.preActivatedValue = layerValue; 
		nextLayer.layerValue = layerValue; 
		nextLayer.layerValue = activate(nextLayer);
		return nextLayer.layerValue;  
	}
}


class TestPropagator extends ForwardPropagator {
	int counter = 0; 
	public double[][] propagate(Layer layer, Layer nextLayer) {
		double[][] testValue; 
		if(counter == 0) layer.testData = appendBiasColumn(layer); counter++; 
		testValue = nt.matrixMultiplication(layer.testData, weightList.get(0));  
		nextLayer.layerValue = testValue; 
		nextLayer.testData = activate(nextLayer);
		return nextLayer.testData; 
	}
	
	public double[][] appendBiasColumn(Layer layer) {
		double[][] layerValue = copyArray(layer.testData);
		double[][] inputsWithBiases = new double[layerValue.length][layerValue[0].length + 1];

		for (int i = 0; i < layerValue.length; i++) {
			for (int j = 0; j < layerValue[0].length; j++) {
				inputsWithBiases[i][j] = layerValue[i][j];
			}
		}
		for (int i = 0; i < layerValue.length; i++) {
			inputsWithBiases[i][layerValue[0].length] = 1;
		}
		return inputsWithBiases;
	}
	
}

class ConvolutionalPropagator extends ForwardPropagator {

}

class PoolingPropagator extends ForwardPropagator {

}
