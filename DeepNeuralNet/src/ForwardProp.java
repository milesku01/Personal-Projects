public class ForwardProp {

	public void CreateLayer(double[][] input) {
		double[][] a = Objects.gtst.MatrixMultiplication(input,
				Objects.gtst.getWeights());
		// System.out.println("a " + java.util.Arrays.deepToString(a));
		double[][] b = Objects.gtst.ApplyTangent(a);
		// System.out.println("b " + java.util.Arrays.deepToString(b));
		double[][] c = Objects.gtst.addBiases(b);
		// System.out.println("c " + java.util.Arrays.deepToString(c));
		Objects.gtst.setLayerOne(c);
	}

	public void CreateSecondLayer(double[][] input) {
		double[][] a = Objects.gtst.MatrixMultiplication(input,
				Objects.gtst.getWeights2());

		double[][] b = Objects.gtst.ApplyTangent(a);
		double[][] c = Objects.gtst.addBiases(b);
		// System.out.println("c " + java.util.Arrays.deepToString(c));
		Objects.gtst.setLayerTwo(c);
	}

	public void CreateResult(double[][] input) {
		double[][] a = Objects.gtst.MatrixMultiplication(input,
				Objects.gtst.getResultWeights());

		double[][] b = Objects.gtst.ApplyTangent(a);
		// System.out.println("b " + java.util.Arrays.deepToString(b));
		// no bias required

		Objects.gtst.setResult(b);
	}
}
