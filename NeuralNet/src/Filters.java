
import java.util.List;

public class Filters {
	int numofFilters; 
	int filterSize;
	int previousDepth; 
	List<double[][]> twoDFilterArray;
	List<double[][][]> threeDFilterArray;
	
	
	public Filters(int numofFilters, int filterSize, int previousDepth) {
		this.numofFilters = numofFilters;
		this.filterSize = filterSize;
		this.previousDepth = previousDepth; 
	}
	
	
}
