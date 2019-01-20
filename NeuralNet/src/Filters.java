import java.util.ArrayList;
import java.util.List;

public class Filters {
	int numofFilters; 
	int filterSize;
	List<double[][]> twoDFilterArray = new ArrayList<double[][]>();
	List<double[][][]> threeDFilterArray = new ArrayList<double[][][]>();
	
	
	public Filters(int numofFilters, int filterSize) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize;
	}
}
