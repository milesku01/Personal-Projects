
public class Trader {
	private double shares = 0; 
	private double balance; 
	
	public Trader(double balance) {
		this.balance = balance; 
	}
	
	public void setBalance(double balance) {
		this.balance = balance;
	}
	
	public double getBalance() {
		return balance; 
	}
	
	public void setShares(double shares) {
		this.shares = shares;
	}
	
	public double getShares() {
		return shares; 
	}
	
	public void addToBalance(double adjust) {
		balance += adjust; 
	}
	
	public void addToShares(double d) {
		shares += d;
	}
	
	public Trade buyAtPrice(double buyPrice, double shares, int time) {
		//if(balance - shares*buyPrice > 0) {
			//this.shares += shares;
			return new Trade(buyPrice, time, shares,"BUY", "VALUE");
		//}
		//System.out.println("Unable to add buy order, costs too much during placed order");
		//return new Trade("NULL"); 
	}
	
	public Trade sellAtPrice(double sellPrice, double shares, int time) {
		//if(this.shares > shares) {
		//	this.shares -= shares;
			return new Trade(sellPrice, time, shares, "SELL", "VALUE");
		//}
		//System.out.println("Unable to add sell order, not enough shares");
		//return new Trade("NULL"); 
	}
	
	public Trade buyAtTime(double buyPrice, double shares, int time) {
		if(balance - shares*buyPrice > 0) {
			this.shares += shares;
			return new Trade(buyPrice, time, shares, "BUY", "TIME");
		}
		System.out.println("Unable to add buy order, costs too much");
		return new Trade("NULL"); 
	}
	
	public Trade sellAtTime(double sellPrice, double shares, int time) {
		if(this.shares > shares) {
			this.shares -= shares;
			return new Trade(sellPrice, time, shares, "SELL", "TIME"); 
		}
		System.out.println("Unable to add sell order, not enough shares");
		return new Trade("NULL"); 
	}
	
}
