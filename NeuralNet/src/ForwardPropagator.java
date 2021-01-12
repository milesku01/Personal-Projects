import java.util.ArrayList;
import java.util.List;

/**
 * Class ForwardPropagator is used to run a forward propagation cycle of the neural network
 * This multiples layers together and activates to generate an output layer 
 *
 */
public class ForwardPropagator {
	static double heldDropoutProb = 0.0;

	static int batchSize = 1;
	static int remainingBatchSize = 0;

	ForwardPropagator forwardPropObj;
	List<ForwardPropagator> propagationObjects;
	List<ForwardPropagator> testPropagationObjects;
	static List<double[][]> weightList = new ArrayList<double[][]>();
	List<Layer> layerList = new ArrayList<Layer>();

	Activator activator = new Activator();

	/**
	 * Runs one instance of a forward propagation (only one batch) Invokes the
	 * propagation method on each sequential pair of layers and updates the values
	 * in the next layer (previousLayer is not updated)
	 * 
	 */
	public void runPropagation() {
		for (int i = 0; i < layerList.size() - 1; i++) {
			propagate(layerList.get(i), layerList.get(i + 1));
		}
	}

	/**
	 * Runs one instance of forward propagation that occurs in the test portion of
	 * the training (to measure accuracy) Invokes the propagation method on each
	 * sequential pair of layers and updates the values in the next layer
	 * (previousLayer is not updated)
	 * 
	 */
	public void runPropagationTest() {
		for (int i = 0; i < layerList.size() - 1; i++) {
			propagateTest(layerList.get(i), layerList.get(i + 1));
		}
	}

	/**
	 * Runs an instance of propagation between two layers (layer and nextLayer) by
	 * calling a list of propagation objects which directs the propagation to occur
	 * in the correct class (polymorphically) Upon creation of each layer Object
	 * it's assigned is a layerPosition which controls which propagation object is
	 * called And then the propagate method from the proper class is used to run the
	 * propagation (matrix multiplication and activation)
	 * 
	 * @param layer     Layer that will not be updated (values of this layer are
	 *                  multiplied with weights to create the next layer)
	 * @param nextLayer assumes the value of the operations of the previous
	 *                  multiplication
	 */
	public void propagate(Layer layer, Layer nextLayer) {
		propagationObjects.get(layer.layerPosition).propagate(layer, nextLayer);
	}

	/**
	 * Much like the propagate method propagateTest runs an instance of propagation
	 * between two layers by calling a list of propagation objects which directs the
	 * propagation to occur in the correct class
	 * 
	 * @param layer     that will not be updated (values of this layer are
	 *                  multiplied with weights to create the next layer)
	 * @param nextLayer assumes the value of the operations of the previous
	 *                  multiplication
	 * 
	 */
	public void propagateTest(Layer layer, Layer nextLayer) {

		testPropagationObjects.get(layer.layerPosition).propagate(layer, nextLayer);

		if (layer instanceof DropoutLayer) {
			((DropoutLayer) layer).dropoutProbability = heldDropoutProb; // clumsily written but should work //TODO: fix
		}
	}

	/**
	 * 
	 * TODO: Rewrite 
	 * constructs the proper polymorphic objects that fit with the type of
	 * propagation first initializes the local weight list from the weight object
	 * and then allocated the memory for the propagation lists
	 * 
	 * Conditionally if the number of sets is above the threshold then an testing
	 * propagation object is immediately to the testPropagation objects which is
	 * then allocated space and then added to the list of test objects
	 * 
	 * then the number of layer pairs is looped through (layerList size - 1) (the
	 * output layer doesn't require a propagator) based on the type of object, the
	 * appropriate propagator type is added to the list
	 * 
	 * conditionally if the number of sets is above the threshold than the object
	 * type is also added to the test propagation list (with the exception of the
	 * first object which is added automatically)
	 * 
	 * 
	 * @param model    the network model from which the layerList is extracted to
	 *                 make propagation objects
	 * @param weights: the list of weights is initialized here because this method
	 *                 is run at the beginning and only run once
	 */
	public void constructForwardPropagationObjects(NetworkModel model, Weights weights) { // only occur once

		layerList = model.layerList;
		weightList = weights.weightList;
		propagationObjects = new ArrayList<ForwardPropagator>(layerList.size());
		
		if (Utility.testSplitThreshold(layerList.get(0))) {
			testPropagationObjects = new ArrayList<ForwardPropagator>(layerList.size());
			forwardPropObj = new TestPropagator();
			testPropagationObjects.add(forwardPropObj);
		}
		

		for (int i = 0; i < layerList.size() - 1; i++) { // minus one because returns last value
			if (Utility.testSplitThreshold(layerList.get(0)) && i > 0) {
				forwardPropObj = new DenseTestPropagator();
				testPropagationObjects.add(forwardPropObj);
			}

			if (layerList.get(i) instanceof HiddenLayer) {
				forwardPropObj = new DensePropagator();
			} else if (layerList.get(i) instanceof InputLayer) {
				if (model.modelType.equals("EVALUATOR")) {
					forwardPropObj = new InputLayerEvaluatorPropagator();
				} else {
					forwardPropObj = new InputLayerPropagator();
				}
			} else if (layerList.get(i) instanceof DropoutLayer) {
				forwardPropObj = new DropoutPropagator();
			}

			propagationObjects.add(forwardPropObj);
		}

	}

	/**
	 * Activates the layerValue by calling an instance of the activator class
	 * 
	 * @param layer: the layer being activated
	 * @return returns the newly activated layer to be assigned to the layerValue of
	 *         the nextLayer
	 */
	public double[][] activate(Layer layer) {
		return activator.activate(layer);
	}

}

/**
 * Class inputLayerPropagator is a subclass of ForwardPropagator and handles the
 * forward propagation between the first layer of the network with the first set
 * of weights. Because of batching this propagator is separated from the
 * standard DensePropagator
 * 
 *
 */
class InputLayerPropagator extends ForwardPropagator {

	int batchTracker = 0;

	/**
	 * Overloaded method propagate assigns the current batch to the currentBatch
	 * member of the inputLayer class (assigned so that the batch can be viewed if
	 * necessary and for back propagation)
	 * 
	 * the preactivated value of the next layer (which is simply the matrix
	 * multiplication of the initial weights and the first layerValue and the first
	 * weight values) is assigned and the nextLayers actual value is assigned by
	 * activating the preactivated value
	 */
	public void propagate(Layer layer, Layer nextLayer) {

		layer.currentBatch = getBatch((InputLayer) layer);
		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.currentBatch,
				weightList.get(layer.layerPosition));
		nextLayer.layerValue = activate(nextLayer);

		
		// //TODO remove
	}

	/**
	 * gets the current batch to be propagated by the propagate method, uses a batch
	 * tracking and batch list system to get them (it gets the first batch and then
	 * increases the counter to get the second batch) if the final batch as been
	 * reached (indicated by a flag in assigned to the batchObject) then the counter
	 * resets to begin the training process for another epoch
	 * 
	 * @param layer: the layer which contains the batch references
	 * @return returns the batch to be operated on / worked with
	 */
	private double[][] getBatch(InputLayer layer) {
		double[][] batch = layer.batchList.get(batchTracker).batchValue;

		if (layer.batchList.get(batchTracker).finalBatch) {
			batchTracker = 0;
		} else {
			batchTracker++;
		}
		return batch;
	}
}

/**
 * Class InputLayerEvaluatorPropagator is a subclass of ForwardPropagator and
 * handles the forward propagation between the first layer of the network with
 * the first set of weights in the case of network evaluation. Separated from
 * the dense propagator for cleanliness (no propagator to object mismatching)
 * 
 *
 */
class InputLayerEvaluatorPropagator extends ForwardPropagator {

	/**
	 * Overloaded method propagate calculates the value of the next layer in
	 * evaluation the preactivated value of the next layer (which is simply the
	 * matrix multiplication of the initial weights and the first layerValue and the
	 * first weight values) is assigned and the nextLayers actual value is assigned
	 * by activating the preactivated value
	 */
	public void propagate(Layer layer, Layer nextLayer) {
		layer.layerValue = Utility.appendBiasColumn(layer.layerValue);
		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.layerValue,
				weightList.get(layer.layerPosition));

		nextLayer.layerValue = activate(nextLayer);
	}

}

/**
 * Class Dense propagator is a subclass of forwardPropagator and handles the
 * forward propagation between any "Dense Layers" which are any standard type of
 * layer that only requires matrix multiplication and activation so this will be
 * the most commonly used propagator
 *
 */
class DensePropagator extends ForwardPropagator {

	/**
	 * Overloaded propagate method assigns assigns the value of the next layer to
	 * the result of the multiplication and activation of the weights and the
	 * previous layer
	 * 
	 * First the "current" layer (previous layer) is given a bias column which will
	 * shift the graph of the equation to better fit the curve then the next layers
	 * preactivated value is set by the matrix multiplication of the previous layer
	 * and the weights then the layer value is created and assigned by activating
	 * that result
	 * 
	 */
	public void propagate(Layer layer, Layer nextLayer) {
		layer.layerValue = Utility.appendBiasColumn(layer.layerValue);

		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.layerValue,
				weightList.get(layer.layerPosition));
		nextLayer.layerValue = activate(nextLayer);

	}
}

/**
 * Class dropoutPropagator is a subclass of ForwardPropagator and handles the
 * forward propagation between a dropout layer and another layer Dropout Layers
 * have some of the neurons (columns) randomly set to 0 which increases the
 * generalization of the network and helps it to find new paths
 * 
 */
class DropoutPropagator extends ForwardPropagator {
	/**
	 * The dropout operation first occurs which randomly zeros columns of the
	 * layerValue using a specified probability Then a bias column is appended to
	 * the layerValue to introduce and element into the network which shifts the
	 * graph directionally and helps the network to generalize then the next layers
	 * preactivated value is set by the matrix multiplication of the previous layer
	 * and the weights then the layer value is created and assigned by activating
	 * that result
	 * 
	 */
	public void propagate(Layer layer, Layer nextLayer) {

		dropOut(layer);
		layer.layerValue = Utility.appendBiasColumn(layer.layerValue);
		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.layerValue,
				weightList.get(layer.layerPosition));
		nextLayer.layerValue = activate(nextLayer);
	}

	/**
	 * The dropout method randomly zeros columns of the layerValue of "layer"
	 * according to the dropout probability This is done to improve the
	 * generalization of the network
	 * 
	 * The reference of layer.layerValue is set to the variable layerValue The
	 * probability is assigned to a variable from the dropoutProbability Using a for
	 * loop each column is cycled through (outer for loop)
	 * 
	 * Then if a random number between 0 and 1 is less than the specified
	 * probability then the next operation occurs (essentially a random number of
	 * columns proportional to the probablity are selected) then if the if condition
	 * is met then each of the values in that column are set to 0 (inner for loop)
	 * 
	 * 
	 * @param layer: The dropout layer to be operated on
	 */
	private void dropOut(Layer layer) {
		double[][] layerValue = layer.layerValue;
		double probability = ((DropoutLayer) layer).dropoutProbability;

		for (int i = 0; i < layerValue[0].length; i++) {
			if (Math.random() <= probability) {
				for (int j = 0; j < layerValue.length; j++) {
					layerValue[j][i] = 0;
				}
			}
		}
	}
}

/**
 * Class testPropagator is a subclass of ForwardPropagator and handles in lieu
 * of the inputPropagator for test cases
 * 
 */
class TestPropagator extends ForwardPropagator {

	/**
	 * The nextLayers preActivated values are created by multiplying the testData of
	 * the layer with the weights Then the nextLayers testData is assigned by
	 * activating the previous result
	 */
	public void propagate(Layer layer, Layer nextLayer) {

		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.testData, weightList.get(layer.layerPosition));

		nextLayer.testData = activate(nextLayer);
	}
}

/**
 * Class DenseTestPropagator is a subclass of ForwardPropagator and handles in lieu
 * of the DensePropagator for test cases
 * 
 */
class DenseTestPropagator extends ForwardPropagator {

	/**
	 * The nextLayers preActivated values are created by multiplying the testData of
	 * the layer with the weights Then the nextLayers testData is assigned by
	 * activating the previous result
	 */
	public void propagate(Layer layer, Layer nextLayer) {
		
		layer.testData = Utility.appendBiasColumn(layer.testData);

		nextLayer.preActivatedValue = Utility.matrixMultiplication(layer.testData, weightList.get(layer.layerPosition));

		nextLayer.testData = activate(nextLayer);
	}
}

