import java.util.ArrayList;
import java.util.List;

public class Gradients {
	double[][] twoDGradient; 
	
	double[][] dRelu;
	double[][] dPool;
	
	double[][][] runningTotal;
	
	List<double[][][]> threeDGradientList = new ArrayList<double[][][]>(); 
	
	String gradientIdentifier;
}
