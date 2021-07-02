
public class Trade {
	double value;
	String buyOrSell;
	String valueOrTime;
	int time;
	double shares; 
	
	public Trade(double value, int time, double d, String buyOrSell, String valueOrTime) {
		this.value = value;
		this.time = time;
		this.shares = d;
		this.buyOrSell = buyOrSell;
		this.valueOrTime = valueOrTime;
	}
	
	public Trade(String buyOrSell) {
		this.buyOrSell = buyOrSell;
	}
	
	public String getBuyOrSell() {
		return buyOrSell; 
	}
	public double getValue() {
		return value; 
	}
	public String getValueOrTime() {
		return valueOrTime; 
	}
	public int getTime() {
		return time; 
	}
	public double getShares() {
		return shares; 
	}
	
	public String toString() {
		return buyOrSell + " Trade for " + shares + " at " + value + " at time " + time;	
	}
}
