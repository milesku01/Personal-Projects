import java.util.ArrayList;
import java.util.List;

public class Gradients {
	double[][] twoDGradient; 
	//double[][][] threeDGradient;
	
	double[][] dRelu;
	double[][] dPool;
	
	double[][][] runningTotal;
	
	//List<double[][]> twoDGradientList = new ArrayList<double[][]>();
	List<double[][][]> threeDGradientList = new ArrayList<double[][][]>(); 
	
	String gradientIdentifier;
}
