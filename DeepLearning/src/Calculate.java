import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;


public class Calculate {
	
	public void calculate(){ // x and y represent inputs in this case
		
		ArrayList<Double> list = new ArrayList<Double>();
		ArrayList<Double> Inputs = new ArrayList<Double>(); 
		BufferedReader br = null; 
		String hiddenLayer;
		double hiddenLayerNum;
		String weight;
		double weightNum; 
		String input;
		double inputNum;
		double x = 0; 
		double y = 0; 
		double[][] InputArray; 
		int counter = 1; 
		
		try{
			br = new BufferedReader(new FileReader("C:\\Users\\Miles\\Desktop\\Inputs.txt"));
			while((input = br.readLine()) != null){ 
				inputNum = Double.parseDouble(input);
				Inputs.add(inputNum); 
			}
			x = Inputs.get(0);
			y = Inputs.get(1); 
			InputArray = new double[(int)x][(int)y]; 
			for(int i=0; i<y ; i++) {
				for(int j=0; j<x; j++){
					counter++;
					InputArray[j][i] = Inputs.get(counter);
				
				}
			}
			br.close(); 
		}
		catch (Exception e) {
			System.out.println("Error occured in inputs");
		}
		try { 
			br = new BufferedReader(new FileReader("C:\\Users\\Miles\\Desktop\\Weights.txt")); 	
			while((weight = br.readLine()) != null) {
				weightNum = Double.parseDouble(weight); 
				list.add(weightNum); 
				
			}
			br.close();
		}
		catch(Exception e) { 
			System.out.println("Possible error in weights");
		}
		
		
		
		double[][] hiddenLayerOne = Object.fp.CreateNode(InputArray, Object.gs.getWeights(), x, y);
		
	}
}
