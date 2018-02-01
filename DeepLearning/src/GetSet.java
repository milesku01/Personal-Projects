public class GetSet { // entire purpose of class is to get set values

	private double[][] randomWeights;
	private double[][] randomWeights2;
	private double[][] randomWeights3;
	private double[][] randomWeights4;
	private double[][] Targets;
	private double[][] inputs;

	public void setHiddenLayer(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getHiddenLayer() {
		return randomWeights;
	}

	public void setHiddenLayer2(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getHiddenLayer2() {
		return randomWeights;
	}

	public void setHiddenLayer3(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getHiddenLayer3() {
		return randomWeights;
	}

	public void setResult(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getResult() {
		return randomWeights;
	}

	// gets and sets weights for x is larger than y
	public void setWeights(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getWeights() {
		return randomWeights;
	}

	public void setWeights2(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getWeights2() {
		return randomWeights;
	}

	public void setWeights3(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getWeights3() {
		return randomWeights;
	}

	public void setWeights4(double[][] randomWeights) {
		this.randomWeights = randomWeights;
	}

	public double[][] getWeights4() {
		return randomWeights;
	}

	void makeWeights(int x, int y) { // makes weights

		double[][] randomWeights = new double[y][x];
		// weights between input and hidden node 1
		for (int k = 0; k < y; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights[k][l] = (double) (Math.random() * 1);
			}
		}

		double[][] randomWeights2 = new double[x][x];
		// weights between hidden node 1 and hidden node 2
		for (int k = 0; k < x; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights[k][l] = (double) (Math.random() * 1);
			}
		}

		double[][] randomWeights3 = new double[x][y];
		// weights between hidden node 3 and hidden node 4
		for (int k = 0; k < x; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights[k][l] = (double) (Math.random() * 1);
			}
		}

		double[][] randomWeights4 = new double[1][x];
		// weights between hidden node 4 and result
		for (int k = 0; k < 1; k++) {
			for (int l = 0; l < x; l++) {
				randomWeights[k][l] = (double) (Math.random() * 1);
			}
		}

		setWeights(randomWeights);
		setWeights2(randomWeights2);
		setWeights3(randomWeights3);
		setWeights4(randomWeights4);

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

	public static double SigmoidFunction(double SynapticAnswer) { // used by the
																	// return
																	// sigmoid
																	// function
		double h;
		h = ((1) / (1 + Math.pow(2.71828, -SynapticAnswer)));
		return h;
	}

	public double[][] returnSigmoid(double[][] a, int x, int y) { // returns the
																	// double[][]
																	// with
																	// sigmoid
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
				sigmoidResult[x][y] = SigmoidFunction(a[x][y]);
			}
		}

		return sigmoidResult;
	}

	public static double[][] MatrixMultiplication(double A[][], double B[][],
			int r1, int c1, int r2, int c2) {

		double[][] product = new double[r1][c2];
		for (int i = 0; i < r1; i++) {
			for (int j = 0; j < c2; j++) {
				for (int k = 0; k < c1; k++) {
					product[i][j] += A[i][k] * B[k][j];
				}
			}
		}
		return product;

	}

}
