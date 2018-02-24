import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;


public class Calculate {
	
	public void calculate(){
		ArrayList<Double> list = new ArrayList<Double>();
		ArrayList<Double> hidden = new ArrayList<Double>(); 
		BufferedReader br = null; 
		String hiddenLayer;
		double hiddenLayerNum;
		String weight;
		double weightNum; 
		try { 
			br = new BufferedReader(new FileReader("C:\\Users\\Miles\\Desktop\\Weights.txt")); 	
			while((weight = br.readLine()) != null) {
				weightNum = Double.parseDouble(weight); 
				list.add(weightNum); 
				br.close();
			}
		}
		catch(Exception e) { 
			System.out.println("Possible error ");
		}
		
		try { 
			br = new BufferedReader(new FileReader("C:\\Users\\Miles\\Desktop\\HiddenLayers.txt")); 	
			while((hiddenLayer = br.readLine()) != null) {
				hiddenLayerNum = Double.parseDouble(hiddenLayer); 
				hidden.add(hiddenLayerNum); 
				br.close();
			}
		}
		catch(Exception e) { 
			System.out.println("Possible error ");
		}
	}
}
