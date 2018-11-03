import java.util.ArrayList;
import java.util.List;

public class NetworkTrainer {
	int batchSize = 0;
	int numofEpochs = 0; 
	List<Layer> layers; 
	List<double[][]> weightList; 
	Activator activator = new Activator(); 
	double[][] currentBatch; 

	public void train(NetworkModel model, Weights weights) {
		batchSize = model.batchSize;
		numofEpochs = model.numofEpochs;
		layers = model.layerList;
		weightList = weights.weightList;
		
		int iterations = calculateNumofBatches() * numofEpochs; 
		
		for(int i = 0; i < iterations; i++ ) {
			forwardPropagation(); 
			//backPropagation(); 
		}
	}
	
	private int calculateNumofBatches() {
		double rawBatchNum = Math.ceil((double)layers.get(0).layerValue.length / (double) batchSize);
		return (int) rawBatchNum;
	}

	public void forwardPropagation() {
		double[][] preActivatedValue; 
		propagateInputLayer(); 
		for(int i=1; i < layers.size() - 1; i++) {
			 appendBiasColumn(layers.get(i));
			 preActivatedValue = matrixMultiplication(layers.get(i).layerValue, weightList.get(i));
			 layers.get(i+1).setLayerValue(preActivatedValue);
			 layers.get(i+1).preActivatedValue = preActivatedValue;
			 layers.get(i+1).setLayerValue(activate(layers.get(i+1)));
			 formatOutput(); 
		 }
	}
	
	public void propagateInputLayer() {
		double[][] preActivatedValue; 
		appendBiasColumn(layers.get(0));
		currentBatch = getBatch(layers.get(0)); 
		preActivatedValue = matrixMultiplication(currentBatch, weightList.get(0)); 
		layers.get(1).setLayerValue(preActivatedValue);
		layers.get(1).preActivatedValue = preActivatedValue;
		layers.get(1).setLayerValue(activate(layers.get(1))); 
	}
	
	public void formatOutput() {
		System.out.println("Batch used " + java.util.Arrays.deepToString(currentBatch));
		for(int i=0; i < layers.size(); i++) {
			System.out.println( "Layer " + i + java.util.Arrays.deepToString(layers.get(i).layerValue));
		}
	}
	
	public void appendBiasColumn(Layer layer){                            
		double[][] layerValue = layer.layerValue; 
		double[][] inputsWithBiases = new double[layerValue.length][layerValue[0].length + 1]; 
		
		for(int i=0; i < layerValue.length; i++ ) {
			for(int j=0; j<layerValue[0].length; j++){
				inputsWithBiases[i][j] = layerValue[i][j];
			}
		}
		for(int i=0; i<layerValue.length; i++) {
			inputsWithBiases[i][layerValue[0].length] = 1;
		}
		
		layer.setLayerValue(inputsWithBiases); 
	}
	
	int batchTracker = 0;  
	int batchCounter = 0; 
	
	public double[][] getBatch(Layer layer) {
		double[][] batch; 
		int remainingBatchSize = (layer.getLayerValue().length % batchSize);
		
		if(!hasReachedEndofBatch()) {
			batch = new double[batchSize][layer.layerSize + 1]; 
			System.out.println(layer.layerValue.length + " layer value length"); 
			
			for(int i=0; i < batchSize; i++) {
				for(int j=0; j < layer.layerSize + 1; j++) {
					batch[i][j] = layer.layerValue[batchTracker][j];
				}
				batchTracker++; 
			}
			batchCounter++; 
			
		} else {
			System.out.println("used remaining batch");
			batch = new double[remainingBatchSize][layer.layerSize + 1];
			
			for(int i=0; i < remainingBatchSize; i++) {
				for(int j=0; j < layer.layerSize + 1; j++) {
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
		if(calculateNumofBatches() - 1 == batchCounter) {
			return true;
		} else {		
			return false; 
		}
	}
	
	public double[][] activate(Layer layer) {
		double[][] activatedValue;
		activatedValue = activator.activate(layer.layerValue, layer.activation); 
		return activatedValue;
	}
	
	public void backPropagation() {
		computeGradients(); 
		updateBiasedFirstMomentEstimate();
		updateBiasedSecondMomentEstimate(); 
		computeBiasCorrectedFirstMoment();
		computeBiasCorrectedSecondMoment();
		updateParameters(); 
	}
	
	double[][] previousPartialGradient; 
	List gradients = new ArrayList();  
	
	private void computeGradients() {
		double[][] gradient; 
		previousPartialGradient = computePartialGradientLastLayer(); 
		gradient = matrixMultiplication(matrixTranspose(layers.get(layers.size()-2).layerValue), previousPartialGradient);
		gradients.add(gradient);
		
		for(int i=weightList.size()-1; i > 0; i--) { //must be edited because hidden layers are propagated differently
			gradient = matrixMultiplication(previousPartialGradient, matrixTranspose(weightList.get(i)));
			gradient = elementwiseMultiplication(gradient, computeDerivative(layers.get(i)));
			gradient = removeBiasColumn(gradient);
			previousPartialGradient = gradient; 
			gradient = matrixMultiplication(matrixTranspose(layers.get(i-1).layerValue), gradient); 
			gradients.add(gradient);  
		}
		
	}
	
	private double[][] computeDerivative(Layer input) {
		double[][] derivative;
		derivative = activator.computeActivatedDerivative(input.preActivatedValue, input.activation);
		return derivative; 
	}
	
	private double[][] removeBiasColumn(double[][] layerValue) {
		double[][] result = new double[layerValue.length][layerValue[0].length - 1]; 
		for(int i=0; i < layerValue.length; i++) {
			for(int j=0; j < layerValue[0].length - 1; j++) {
				result[i][j] = layerValue[i][j];
			}
		}
		return result;
	}
	
	private double[][] computePartialGradientLastLayer() {
		
		return null;
	}

	private double[][] matrixTranspose(double[][] m){
	        double[][] temp = new double[m[0].length][m.length];
	        for (int i = 0; i < m.length; i++)
	            for (int j = 0; j < m[0].length; j++)
	                temp[j][i] = m[i][j];
	      return temp;
	}
	
	private void updateBiasedFirstMomentEstimate() {
		// TODO Auto-generated method stub
		
	}

	private void updateBiasedSecondMomentEstimate() {
		// TODO Auto-generated method stub
		
	}

	private void computeBiasCorrectedFirstMoment() {
		// TODO Auto-generated method stub
		
	}

	private void computeBiasCorrectedSecondMoment() {
		// TODO Auto-generated method stub
		
	}

	private void updateParameters() {
		// TODO Auto-generated method stub
		
	}

	private double[][] matrixMultiplication(double[][] A, double[][] B) {
		 
			int aRows = A.length;
			int aColumns = A[0].length;
			int bRows = B.length;
			int bColumns = B[0].length;
			double[][] C = new double[aRows][bColumns];

			if (aColumns != bRows) {
				throw new IllegalArgumentException("A:col: " + aColumns
						+ " did not match B:rows " + bRows + ".");
			}

			for (int i = 0; i < aRows; i++) { // aRow
				for (int j = 0; j < bColumns; j++) { // bColumn
					for (int k = 0; k < aColumns; k++) { // aColumn
						C[i][j] += A[i][k] * B[k][j];
					}
				}
			}
			return C;
		}

}
