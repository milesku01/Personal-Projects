public class GetSet { // entire purpose of class is to get set values

	private double[][] randomWeights;
	private double[][] randomWeights2;
	private double[][] randomWeights3;
	private double[][] randomWeights4;
	private double[][] randomWeightsInsert; 
	private double[][] deltaOutputSum; 
	private double[][] hiddenLayer3PreSigmoid; 
	private double[][] resultPreSigmoid; 
	private double[][] hl; 
	private double[][] hl2; 
	private double[][] hl3; 
	private double[][] hlInsert; 
	private double[][] result; 
	private double[][] Targets;
	private double[][] inputs;

	
	public void setHiddenLayer(double[][] randomWeights) {
		hl = randomWeights;
	}

	public double[][] getHiddenLayer() {
		return hl;
	}

	public void setHiddenLayer2(double[][] randomWeights) {
		hl2 = randomWeights;
	}

	public double[][] getHiddenLayer2() {
		return hl2;
	}

	public void setHiddenLayer3(double[][] randomWeights) {
		hl3 = randomWeights;
	}

	public double[][] getHiddenLayer3() {
		return hl3;
	}

	public void setResult(double[][] randomWeights) {
	 result = randomWeights;
	}

	public double[][] getResult() {
		return result;
	}
	
	public void setHiddenLayerInsert(double[][] randomWeights) {
		hlInsert = randomWeights;
	}

	public double[][] getHiddenLayerInsert() {
		return hlInsert;
	}


	// gets and sets weights for x is larger than y
	
	public void setWeightsInsert(double[][] randomWeights) {
		this.randomWeightsInsert = randomWeights;
	}

	public double[][] getWeightsInsert() {
		return randomWeightsInsert;
	}
	
	
	public void setWeights(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getWeights() {
		return randomWeights;
	}

	public void setWeights2(double[][] randomWeights) {
		this.randomWeights2 = randomWeights;
	}

	public double[][] getWeights2() {
		return randomWeights2;
	}

	public void setWeights3(double[][] randomWeights) {
		this.randomWeights3 = randomWeights;
	}

	public double[][] getWeights3() {
		return randomWeights3;
	}

	public void setWeights4(double[][] randomWeights) {
		this.randomWeights4 = randomWeights;
	}

	public double[][] getWeights4() {
		return randomWeights4;
	}
	public void setHiddenLayer3PreSigmoid(double[][] hiddenLayer3PreSigmoid) { 
		this.hiddenLayer3PreSigmoid = hiddenLayer3PreSigmoid; 
	}
	public double[][] getHiddenLayer3PreSigmoid(){
		return hiddenLayer3PreSigmoid; 
	}
	public void setResultPreSigmoid(double[][] resultPreSigmoid){
		this.resultPreSigmoid = resultPreSigmoid; 
	}
	public double[][] getResultPreSigmoid() { 
		return resultPreSigmoid; 
	}

	void makeWeights(int x, int y) { // makes weights

		double[][] randomWeights = new double[y][x];
		// weights between input and hidden node 1
		for (int k = 0; k < y; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights[k][l] = (double) ((Math.random() * 2)-1);
			}
		}

		double[][] randomWeights2 = new double[y][x];
		// weights between hidden node 1 and hidden node 2
		for (int k = 0; k < y; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights2[k][l] = (double) ((Math.random() * 2)-1);
			}
		}

		double[][] randomWeights3 = new double[x][y];
		// weights between hidden node 3 and hidden node 4
		for (int k = 0; k < x; k++) {
			for (int l = 0; l < y; l++) {
				randomWeights3[k][l] = (double) ((Math.random() * 2)-1);
			}
		}

		double[][] randomWeights4 = new double[1][x];
		// weights between hidden node 4 and result
		for (int k = 0; k < 1; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights4[k][l] = (double) ((Math.random() * 2)-1);
			}
		}
		
		double[][] randomWeightsInsert = new double[x][y];
		// weights between hidden node 4 and result
		for (int k = 0; k <x ; k++) {
			for (int l = 0; l <y ; l++) {
				randomWeightsInsert[k][l] = (double) ((Math.random() * 2)-1);
			}
		}

		setWeights(randomWeights);
		setWeights2(randomWeights2);
		setWeights3(randomWeights3);
		setWeights4(randomWeights4);
		setWeightsInsert(randomWeightsInsert); 

	}

	public void setInputs(double[][] inputs) {
		this.inputs = inputs;
	}

	public void setTarget(double[][] targetMatrix) {
		this.Targets = targetMatrix;
	}

	public double[][] getInputs() {
		return inputs;
	}

	public double[][] getTarget() {
		return Targets;
	}
	
	//next getters and setters are for backPropagation
	
	public void setDeltaOutputSum(double[][] deltaOutputSum){
		this.deltaOutputSum = deltaOutputSum; 
	}
	public double[][] getDeltaOutputSum() {
		return deltaOutputSum;
	}
	
	
	public double[][] specializedDivide(double[][] A, double[][] B) { 
		// B is divided by A 
		double[][] answer =null;
		
		
		
		return answer; 
	}
	
	

	public static double SigmoidFunction(double SynapticAnswer) { // used by the
																	// return
																	// sigmoid
																	// function
		double h;
		h = ((1) / (1 + Math.pow(2.71828, -SynapticAnswer)));
		return h;
	}

	public double[][] returnSigmoid(double[][] a) { // returns the
																	// double[][]
																	// with
		int x=a.length; 
		int y=a[0].length; 															// sigmoid
																	// function
																	// applied
																	// to each
																	// number in
																	// the set

		double[][] sigmoidResult = new double[x][y]; // a is the matrix
														// calculated by
														// multiplying weights1
														// and inputs

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < y; j++) {
				sigmoidResult[i][j] = SigmoidFunction(a[i][j]);
			}
		}

		return sigmoidResult;
	}

	 public double[][] MatrixMultiplication(double[][] A, double[][] B) {

	        int aRows = A.length;
	        int aColumns = A[0].length;
	        int bRows = B.length;
	        int bColumns = B[0].length;
	         
	        if (aColumns != bRows) {
	            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
	        }
	
	        double[][] C = new double[aRows][bColumns];
	        for (int i = 0; i < aRows; i++) {
	            for (int j = 0; j < bColumns; j++) {
	                C[i][j] = 0.00000;
	            }
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
	
	 //need to initialize
	 public double[][] AddAcross(double[][] a, double[][] b){
		 
		 double[][] result = new double[a.length][a[0].length]; 
		 
		 if (a.length != b.length || a[0].length != b[0].length){
			 throw new IllegalArgumentException("Input did not match, arrays must be same dimensions; Subtraction Method");
		 }
		 
		 for(int i =0; i<a.length; i++) { 
			 for(int j=0; j<a[0].length; j++){
				 result[i][j] = a[i][j] - b[i][j]; 
			 }
		 }
		 
		 return result;
	 }
	 public double[][] MultiplyAcross(double[][] a, double[][] b){ 
		 
		 double[][] result = new double[a.length][a[0].length]; 
		 
		 if (a.length != b.length || a[0].length != b[0].length){
			 throw new IllegalArgumentException("Input did not match, arrays must be same dimensions; Multiplication Method");
		 }
		 
		 for(int i =0; i<a.length; i++) { 
			 for(int j=0; j<a[0].length; j++){
				 result[i][j] = a[i][j] * b[i][j]; 
			 }
		 }
		 
		 return result; 
	 }
	 
	 //these two methods apply a derivative digmoid to every number in an array
	 public double[][] ApplySigmoidDerivative(double[][] a) { // , instead of ;

			// instead of int sum = new sum[a.length];
			double[][] sum = new double[a.length][a[0].length];

			// i < a.length isntead of a.length
			for (int i = 0; i < a.length; i++) {
				for(int j=0; j<a[0].length; j++) {
				sum[i][j] = ApplyActivationDerivative(a[i][j]);
			}
		}
			return sum;
		}

		private static double ApplyActivationDerivative(double f) {
			double a;
			a = ((Math.pow(2.71828, f)) / (Math.pow(1 + Math.pow(2.71828, f), 2)));
			return a;
		}
		
		public double[][] MatrixTranspose(double [][] m){
		        double[][] temp = new double[m[0].length][m.length];
		        for (int i = 0; i < m.length; i++)
		            for (int j = 0; j < m[0].length; j++)
		                temp[j][i] = m[i][j];
		        return temp;
		    }
	 
}
