public class BackProp {

	public void runIteration(int numofSets, int counter) {
		double[][] weightChange3; // use this for weights 3
		double[][] weightChange2;
		double[][] weightChange;

		double[][] gradientChange4;
		double[][] gradientChange3;
		double[][] gradientChange2;

		double[][] loss;
		double[][] basicLoss;
		double basicLossVal; 
		double lossVal = 0.0;
		double gradientVar = 0.0001;
		double learningRate = .1;
//			double learningRate = (.5/Math.sqrt(Math.sqrt((double)counter)));
		// double learningRate = .1/(1.0 + ((double)counter/500));

		double var = 1.0 / (double)numofSets;
		double adjustment = learningRate * var;

		basicLoss = Objects.gtst.SubtractAcross(Objects.gtst.getResult(),
				Objects.gtst.getTarget());
		basicLoss = Objects.gtst.absoluteValAllElements(basicLoss); 
		basicLossVal = Objects.gtst.sumAllElements(basicLoss); 
		basicLossVal /= (double)numofSets; 
		
		Objects.gtst.setLossBasic(basicLossVal);
		
		loss = Objects.gtst.SubtractAcross(Objects.gtst.getResult(),
				Objects.gtst.getTarget());
		loss = Objects.gtst.MultiplyAcross(loss, loss);
		loss = Objects.gtst.scalarMultiply(loss, .5);
		lossVal = Objects.gtst.sumAllElements(loss) / (double) numofSets;

		double regularizeLoss = (gradientVar / (2.0))
				* (Objects.gtst.sumAllElements(Objects.gtst.MultiplyAcross(
						Objects.gtst.getWeights(), Objects.gtst.getWeights()))
						+ Objects.gtst.sumAllElements(Objects.gtst
								.MultiplyAcross(Objects.gtst.getWeights2(),
										Objects.gtst.getWeights2())) + Objects.gtst
							.sumAllElements(Objects.gtst.MultiplyAcross(
									Objects.gtst.getResultWeights(),
									Objects.gtst.getResultWeights())));

		Objects.gtst.setLoss((lossVal + regularizeLoss));

		gradientChange4 = Objects.gtst.SubtractAcross(Objects.gtst.getTarget(),
				Objects.gtst.getResult());

		double[][] gradientChangePrime4 = Objects.gtst.MultiplyAcross(
				gradientChange4,
				Objects.gtst.scalarMultiply(oneMinusTangent(Objects.gtst
						.ApplyInverseTangent(Objects.gtst.getResult())), -1)); // this
																				// var
																				// used
																				// in
																				// next
																				// weights

		gradientChange4 = (Objects.gtst // change
				.MatrixMultiplication(Objects.gtst.MatrixTranspose(Objects.gtst
						.getLayerTwo()), gradientChangePrime4)); // use
																				// when
																				// adjusting
																				// weights
		// System.out.println(java.util.Arrays.deepToString(gradientChange4));
		// //System check, logic good

		double[][] gradientChangePrime3 = Objects.gtst.MultiplyAcross(
				Objects.gtst.MatrixMultiplication(gradientChangePrime4,
						Objects.gtst.MatrixTranspose(Objects.gtst
								.getResultWeights())),
				oneMinusTangent(Objects.gtst.ApplyInverseTangent(Objects.gtst
						.getLayerTwo())));
		
		gradientChange3 = (Objects.gtst.MatrixMultiplication(Objects.gtst
						.MatrixTranspose(Objects.gtst.getLayerOne()),
						removeLastColumn(gradientChangePrime3))); // use when adjusting weights
		// System.out.println(java.util.Arrays.deepToString(gradientChange3));
		// //System check, looks good

		double[][] gradientChangePrime2 = Objects.gtst.MultiplyAcross(
				Objects.gtst.MatrixMultiplication(
						removeLastColumn(gradientChangePrime3), Objects.gtst
								.MatrixTranspose(Objects.gtst.getWeights2())),
				oneMinusTangent(Objects.gtst.ApplyInverseTangent(Objects.gtst
						.getLayerOne())));
		
		gradientChange2 = (Objects.gtst // change
				.MatrixMultiplication(
						Objects.gtst.MatrixTranspose(Objects.gtst.getInputs()),
						removeLastColumn(gradientChangePrime2)));
		// System.out.println(java.util.Arrays.deepToString(gradientChange2));
		// //System check, looks good

		weightChange3 = Objects.gtst.SubtractAcross(Objects.gtst.getResultWeights(), Objects.gtst
				.scalarMultiply(Objects.gtst.getResultWeights(), gradientVar));
		weightChange2 = Objects.gtst.SubtractAcross(Objects.gtst.getWeights2(), Objects.gtst
				.scalarMultiply(Objects.gtst.getWeights2(), gradientVar));
		weightChange = Objects.gtst.SubtractAcross(Objects.gtst.getWeights(), Objects.gtst
				.scalarMultiply(Objects.gtst.getWeights(), gradientVar)); 
		
		
		
		
		weightChange3 = Objects.gtst.SubtractAcross(
				weightChange3,
				Objects.gtst.scalarMultiply(gradientChange4, adjustment));
		weightChange2 = Objects.gtst.SubtractAcross(weightChange2,
				Objects.gtst.scalarMultiply(gradientChange3, adjustment));
		weightChange = Objects.gtst.SubtractAcross(weightChange,
				Objects.gtst.scalarMultiply(gradientChange2, adjustment));

		Objects.gtst.setResultWeights(weightChange3);
		Objects.gtst.setWeights2(weightChange2);
		Objects.gtst.setWeights(weightChange);

	}

	public double[][] oneMinusTangent(double[][] z) {
		int a = z.length;
		int b = z[0].length;
		double[][] result = new double[a][b];

		for (int i = 0; i < a; i++) {
			for (int j = 0; j < b; j++) {
				result[i][j] = (1 - (Math.pow(Math.tanh(z[i][j]), 2)));
			}
		}

		return result;
	}

	public double[][] removeLastColumn(double[][] a) {

		int x = a.length;
		int y = a[0].length;

		double[][] result = new double[x][y - 1];

		for (int i = 0; i < x; i++) {
			for (int j = 0; j < (y - 1); j++) {
				result[i][j] = a[i][j];
			}
		}
		return result;
	}
}
