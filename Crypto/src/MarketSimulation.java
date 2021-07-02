import java.util.*;

public class MarketSimulation {

	public static void main(String[] args) {
		// import market data, for testing into one array
		// test trading ability

		double[] marketData = inputMarketData("C:\\Users\\kuhnm\\Desktop\\BTC\\BTCMONTH4.txt");
		//double[] marketData = inputMarketData("C:\\Users\\kuhnm\\Desktop\\BasicMarketData.txt");
		
		marketData = reverseArray(marketData);

		Trader trader = new Trader(1000000);

		Strategy basicStrategy = new Strategy("BASIC");

		simulateRealTimeMarket(marketData, trader, basicStrategy);

		// simulateRealTimeMarketLoop(marketData, basicStrategy);

		//printFinalTraderStatistics(trader, marketData[marketData.length - 1]);

	//	simulateRealTimeMarketAll(); // rename
		
		simulateRealTimeMarketAllPercent(); 

	}

	public static void printDataPoints(double[] array) {
		for (int i = 0; i < array.length; ++i) {
			// System.out.println("("+(100*i) + "," +array[i]+")");
			// System.out.println("("+(100*i+1000) + "," +
			// Strategy.calculateStrdDev(Strategy.getArraySegment(array, i, i+8),
			// Strategy.calculateMean(Strategy.getArraySegment(array, i, i+8))) + ")");
			// System.out.println("("+(100*i+1000) + "," +
			// Strategy.calculateSlopeOfBestFit(Strategy.getArraySegment(array,
			// i,i+8))+")");
			// if(i>49) {
			// System.out.println("("+(100*i) + "," +
			// Strategy.calculateMean(Strategy.getArraySegment(array, i-50, i))+")");
			// }
		}
	}

	// test method
	public static double[] reverseArray(double[] array) {
		double[] newArray = new double[array.length];
		for (int i = 0; i < array.length; i++) {
			newArray[i] = array[array.length - i - 1];
		}
		return newArray;
	}

	public static void printFinalTraderStatistics(Trader trader, double finalPrice) {
		System.out.println(trader.getShares());
		System.out.println("Trader Balance " + trader.getBalance());
		System.out.println("Money in coin/stock " + trader.getShares() * finalPrice);
		System.out.println("Value of total assets " + (trader.getBalance() + trader.getShares() * finalPrice));
	}

	public static double[] inputMarketData(String fileName) {
		Parser parser = new Parser(fileName);
		parser.initialize();

		return parser.readFileIntoArray();
	}

	public static void printData(double[] data) {
		for (double num : data) {
			System.out.print(num + " ");
		}
		System.out.println();
	}

	public static void simulateMarket(List<Trade> tradeList, double[] marketData, Trader trader, double buyPercent,
			double sellPercent) { // add for more
		// traders //

		// traders

		// System.out.println(tradeList.size());
		for (int time = 0; time < marketData.length; ++time) { // time loop
			if (!tradeList.isEmpty() && tradeList.get(0).getTime() <= time) {
				for (int i = 0; i < tradeList.size(); ++i) { // modularize later, and add time driven trades
					Trade trade = tradeList.get(i);
					if (time >= trade.getTime()) {
						// Process Buy order
						if (trade.getValueOrTime().equals("VALUE")) {
							if (trade.getBuyOrSell().equals("BUY")) {
								if (executeValueBuy(marketData, trader, trade, time)) {
									tradeList.remove(i); // possible edit these to all happen at same time
									--time; // check for another trade happening at the same time
								}
							} else if (trade.getBuyOrSell().equals("SELL")) { // sell order
								if (executeValueSell(marketData, trader, trade, time)) {
									tradeList.remove(i);
									--time; // check for another trade happening at the same time
								}
							}
						} else if (trade.getValueOrTime().equals("TIME")) {
							if (trade.getBuyOrSell().equals("BUY")) {
								executeTimeBuy(marketData, trader, trade, time);
									tradeList.remove(i);
									--time; // check for another trade happening at the same time
								
							} else if (trade.getBuyOrSell().equals("SELL")) { // sell order
								executeTimeSell(marketData, trader, trade, time);
									tradeList.remove(i);
									--time; // check for another trade happening at the same time
							
							}
						} else if (trade.getValueOrTime().equals("BALANCE")) {
							if (trade.getBuyOrSell().equals("BUY")) {
								if (executeBalanceBuy(marketData, trader, trade, buyPercent, time)) {
									tradeList.remove(i);
									--time; // check for another trade happening at the same time
								}
							} else if (trade.getBuyOrSell().equals("SELL")) { // sell order
								if (executeBalanceSell(marketData, trader, trade, sellPercent, time)) {
									tradeList.remove(i);
									--time; // check for another trade happening at the same time
								}
							}
						}
					}
				}
			}
		}
		// printFinalTraderStatistics(trader, marketData[marketData.length - 1]);
	}

	public static void simulateRealTimeMarket(double[] marketData, Trader trader, Strategy strategy) {
		List<Trade> tradeList = new ArrayList<Trade>(); // local trade list
		// List<Double> slopeList = new ArrayList<Double>();

		for (int time = 0; time < marketData.length; time += 12) { // change to reflect strategy times

			Trade trade = strategy.percentAverageTrading(trader, marketData, time, 12);

			if (!trade.getBuyOrSell().equals("NULL")) {
				tradeList.add(trade);
			}
		}

		printTrades(tradeList);

		simulateMarket(tradeList, marketData, trader, .05, .15);
	}

	public static void simulateRealTimeMarketLoop(double[] marketData, Strategy strategy) {
		// int counter = 0;
		long startTime = 0;
		long endTime = 0;
		long combined = 0;
		double highestTotalValue = 0;
		double BTCRelativeValue = 55641 / marketData[0];
		// 55641 / 1924;
		List<Trade> tradeList = new ArrayList<Trade>(); // local trade list
		// List<Double> slopeList = new ArrayList<Double>();
		Trader trader = new Trader(1000000);

		for (double i = 4 * BTCRelativeValue; i < 12 * BTCRelativeValue; i += 1 * BTCRelativeValue) {
			for (double j = 5 * BTCRelativeValue; j < 12 * BTCRelativeValue; j += 1 * BTCRelativeValue) {
				for (double k = .99; k <= 1; k += .005) {
					for (double l = 1.00; l < 1.14; l += .01) {
						for (double m = 0; m <= .03; m += .005) {
							for (double n = 0; n <= .03; n += .005) {

								// ++counter;
								startTime = System.nanoTime();
								for (int time = 0; time < marketData.length; ++time) {
									Trade trade = strategy.slopeTradingLoop(trader, marketData, time, 6, i, j, k, l, m,
											n);

									if (!trade.getBuyOrSell().equals("NULL")) {
										tradeList.add(trade);
									}
								}
								// System.out.println(i+ " " + j + " " + k+ " " + l+ " " + m + " " + n);
								// simulateMarket(tradeList, marketData, trader);
								endTime = System.nanoTime();
								combined += (endTime - startTime);

								if (highestTotalValue < trader.getBalance()
										+ trader.getShares() * marketData[marketData.length - 1]) {
									highestTotalValue = trader.getBalance()
											+ trader.getShares() * marketData[marketData.length - 1];
								}
								tradeList.clear();
								trader.setBalance(1000000);
								trader.setShares(0);
							}
						}
					}
				}
			}
		}

		System.out.println(combined / 1000000000.0 + " seconds");
		System.out.println("Standard increase " + marketData[marketData.length - 1] / marketData[0]);
		System.out.println(highestTotalValue);
		// System.out.println(i + " " + j + " " + k + " " + l + " " + m + " " + n);
	}

	public static void simulateRealTimeMarketAllRAM() {
		int counter = 0;
		int months = 5;
		int monthsLower = 1;
		long startTime = 0;
		long endTime = 0;
		long combined = 0;

		double valueCounter = 0;
		double BTCRelativeValue = // 55641 / marketData[0];
				1;
		List<Trade> tradeList = new ArrayList<Trade>(); // local trade list
		// List<Double> slopeList = new ArrayList<Double>();
		Trader trader = new Trader(1000000);
		Strategy strategy = new Strategy("BASIC");

		List<Double> dataList = new ArrayList<Double>();

		// yes this method is inefficient, change in production
		for (int month = months; month >= monthsLower; --month) {
			double[] data = inputMarketData("C:\\Users\\kuhnm\\Desktop\\BTC\\BTCMONTH" + month + ".txt");
			data = reverseArray(data);
			for (int i = 0; i < data.length; ++i) {
				dataList.add(data[i]);
			}

			double[] marketData = new double[dataList.size()];

			for (int i = 0; i < marketData.length; ++i) {
				marketData[i] = dataList.get(i);
			}
			
			dataList.clear(); 
			
			

			double highestTotalValue = 0;
			long slopeTradeCheckStart = 0;
			long slopeTradeCheckEnd = 0;
			long slopeTradeCombine = 0;
			long marketCheckStart = 0;
			long marketCheckEnd = 0;
			long marketCombine = 0;

			double storedI = 0;
			double storedJ = 0;
			double storedK = 0;
			double storedL = 0;
			double storedM = 0;
			double storedN = 0;

			startTime = System.nanoTime();
			for (double i = 1; i < 10; i += 1) {
				for (double j = 1; j < 10; j += 1) {
					for (double k = .95; k <= 1; k += .005) {
						for (double l = 1.08; l < 1.14; l += .01) {
							for (double m = .98; m <= 1; m += .01) {
								for (double n = 1.08; n <= 1.12; n += .01) {

									++counter;

									for (int time = 0; time < marketData.length; time += 6) {
										slopeTradeCheckStart = System.nanoTime();
										Trade trade = strategy.slopeTradingLoop(trader, marketData, time, 6, i, j, k,
												l, m, n);
										slopeTradeCheckEnd = System.nanoTime();

										slopeTradeCombine += slopeTradeCheckEnd - slopeTradeCheckStart;

										if (!trade.getBuyOrSell().equals("NULL")) {
											tradeList.add(trade);
										}
									}
									// System.out.println(i+ " " + j + " " + k+ " " + l+ " " + m + " " + n);
									marketCheckStart = System.nanoTime();
									
									//System.out.println(tradeList.size());
									if (tradeList.size() < 30) { // limits trades per month
										simulateMarket(tradeList, marketData, trader, i, j);
									}
									marketCheckEnd = System.nanoTime();

									marketCombine += marketCheckEnd - marketCheckStart;

									if (highestTotalValue < trader.getBalance()
											+ trader.getShares() * marketData[marketData.length - 1]) {
										highestTotalValue = trader.getBalance()
												+ trader.getShares() * marketData[marketData.length - 1];
										// System.out.println(highestTotalValue);
										// System.out.println(i + " " + j + " " + k + " " + l + " " + m + " " + n);
										storedI = i;
										storedJ = j;
										storedK = k;
										storedL = l;
										storedM = m;
										storedN = n;
									}

									tradeList.clear();
									trader.setBalance(1000000);
									trader.setShares(0);
								}
							}
						}
					}
				}
			}

			endTime = System.nanoTime();
			combined += (endTime - startTime);

		//	System.out.println(combined / 1000000000.0 + " seconds");
		//	System.out.println(slopeTradeCombine / 1000000000.0 + " seconds");
		//	System.out.println(marketCombine / 1000000000.0 + " seconds");
			// System.out.println("Standard increase " + marketData[marketData.length - 1] /
			// marketData[0]);
			//System.out.println(counter);

			System.out.println("Month " + month + " summary " + highestTotalValue);
			System.out.println(storedI + " " +storedJ + " " +storedK + " " +storedL + " " +storedM + " " +storedN);
			
		}
	}
	
	public static void simulateRealTimeMarketAllPercent() {
		int counter = 0;
		int months = 10;
		int monthsLower = 1;
		long startTime = 0;
		long endTime = 0;
		long combined = 0;

		List<Trade> tradeList = new ArrayList<Trade>(); // local trade list
		// List<Double> slopeList = new ArrayList<Double>();
		Trader trader = new Trader(1000000);
		Strategy strategy = new Strategy("BASIC");

		List<Double> dataList = new ArrayList<Double>();

		// yes this method is inefficient, change in production
		for (int month = months; month >= monthsLower; --month) {
			double[] data = inputMarketData("C:\\Users\\kuhnm\\Desktop\\BTC\\BTCMONTH" + month + ".txt");
			data = reverseArray(data);
			for (int i = 0; i < data.length; ++i) {
				dataList.add(data[i]);
			}

			double[] marketData = new double[dataList.size()];

			for (int i = 0; i < marketData.length; ++i) {
				marketData[i] = dataList.get(i);
			}
			
			dataList.clear(); 
			
			

			double highestTotalValue = 0;
			long slopeTradeCheckStart = 0;
			long slopeTradeCheckEnd = 0;
			long slopeTradeCombine = 0;
			long marketCheckStart = 0;
			long marketCheckEnd = 0;
			long marketCombine = 0;

			double storedI = 0;
			double storedJ = 0;
			double storedK = 0;
			double storedL = 0;
			double storedM = 0;
			

			startTime = System.nanoTime();
			for (int i = 5; i < 50; i += 1) {
				for (double j = .90; j < 1.00; j += .01) {
					for (double k = 1; k <= 1.5; k += .01) {
					//	for(double l = 2; l < 10; l += 1) {
							//for(double m = 2; m < 10; m += 1) {
						
									++counter;

									for (int time = 0; time < marketData.length; time += 6) {
										slopeTradeCheckStart = System.nanoTime();
										Trade trade = strategy.percentAverageTradingLoop(trader, marketData, time, i, j, k, 5, 5);
										slopeTradeCheckEnd = System.nanoTime();

										slopeTradeCombine += slopeTradeCheckEnd - slopeTradeCheckStart;

										if (!trade.getBuyOrSell().equals("NULL")) {
											tradeList.add(trade);
										}
									}
									// System.out.println(i+ " " + j + " " + k+ " " + l+ " " + m + " " + n);
									marketCheckStart = System.nanoTime();
									
									//System.out.println(tradeList.size());
									if (tradeList.size() < 30) { // limits trades per month
										simulateMarket(tradeList, marketData, trader, i, j);
									}
									marketCheckEnd = System.nanoTime();

									marketCombine += marketCheckEnd - marketCheckStart;

									if (highestTotalValue < trader.getBalance()
											+ trader.getShares() * marketData[marketData.length - 1]) {
										highestTotalValue = trader.getBalance()
												+ trader.getShares() * marketData[marketData.length - 1];
										// System.out.println(highestTotalValue);
										// System.out.println(i + " " + j + " " + k + " " + l + " " + m + " " + n);
										storedI = i;
										storedJ = j;
										storedK = k;
									//	storedL = l;
									//	storedM = m; 
										
									}

									tradeList.clear();
									trader.setBalance(1000000);
									trader.setShares(0);
								}
							}
						}
				//}
		//	}
				
			endTime = System.nanoTime();
			combined += (endTime - startTime);

			System.out.println(combined / 1000000000.0 + " seconds");
			System.out.println(slopeTradeCombine / 1000000000.0 + " seconds");
			System.out.println(marketCombine / 1000000000.0 + " seconds");
			// System.out.println("Standard increase " + marketData[marketData.length - 1] /
			// marketData[0]);
			//System.out.println(counter);

			System.out.println("Month " + month + " summary " + highestTotalValue);
			System.out.println(storedI + " " +storedJ + " " +storedK + " " + storedL + " " + storedM);
			
		}
	}

	public static boolean executeValueBuy(double[] marketData, Trader trader, Trade trade, int time) {

		if (trade.getValue() >= marketData[time]) { // overload operator later
			if (-marketData[time] * trade.getShares() + trader.getBalance() >= 0) { // if has enough in balance to buy

				// System.out.println("Bought at price: " + marketData[time] + " at time " + time);
				trader.addToBalance(-marketData[time] * trade.getShares());
				trader.addToShares(trade.getShares());

			
				return true;
			} else {
				 //System.out.println("Insufficient funds to cover buy " + trade);
			}
		} else {
			 //System.out.println("Cost too high to buy " + trade);
		}
		return false;
	}

	public static boolean executeValueSell(double[] marketData, Trader trader, Trade trade, int time) {

		if (trade.getValue() <= marketData[time]) {
			if (trader.getShares() >= trade.getShares()) { // if has enough shares to sell

				// System.out.println("Sold at price: " + marketData[time] + " at time " +
				// time);
				trader.addToBalance(marketData[time] * trade.getShares());
				trader.addToShares(-trade.getShares());

				
				return true;
			} else {
				//System.out.println("Insufficient number of shares to sell " + trade + " " +
				// time);
			}
		} else {
			// System.out.println("Cost too low to sell " + trade);
		}
		return false;
	}

	public static boolean executeTimeBuy(double[] marketData, Trader trader, Trade trade, int time) {

		if (-marketData[time] * trade.getShares() + trader.getBalance() >= 0) { // if has enough in balance to buy
		//	 System.out.println("Bought at price: " + marketData[time] + " at time " +
			// time);
			trader.addToBalance(-marketData[time] * trade.getShares());
			trader.addToShares(trade.getShares());
			return true;
		} else {
			// System.out.println("Insufficient funds to cover buy " + trade);
		}
		return false;
	}

	public static boolean executeTimeSell(double[] marketData, Trader trader, Trade trade, int time) {
		if (trader.getShares() >= trade.getShares()) { // if has enough shares to sell
		//	 System.out.println("Sold at price: " + marketData[time] + " at time " +
			// time);
			trader.addToBalance(marketData[time] * trade.getShares());
			trader.addToShares(-trade.getShares());
			return true;
		} else {
			// System.out.println("Insufficient number of shares to sell " + trade + " " +
			// time);
		}
		return false;
	}

	public static boolean executeBalanceBuy(double[] marketData, Trader trader, Trade trade, double percent, int time) {

		if (trade.getValue() >= marketData[time]) {
			 if (-marketData[time] * trade.getShares() + trader.getBalance() >= 0) { // if has enough in balance to buy
			 System.out.println("Bought " + percent * trader.getBalance() /
			 marketData[time] + " shares at price: "
			 + marketData[time] + " at time " + time);
			trader.addToBalance(-percent * trader.getBalance());
			trader.addToShares(percent * trader.getBalance() / marketData[time]);
			return true;
		}
		 } else {
		 System.out.println("Insufficient funds to cover buy " + trade);
		 }
		return false;
	}

	public static boolean executeBalanceSell(double[] marketData, Trader trader, Trade trade, double percent,
			int time) {
		if (trade.getValue() <= marketData[time]) {
			if (percent * trader.getShares() > 0) { // if has enough shares to sell
				 System.out.println("Sold " + percent * trader.getShares() + " shares at price: " + marketData[time] + " at time " + time);
				trader.addToBalance(percent * trader.getShares() * marketData[time]); // can make this based on balance
																						// too
				trader.addToShares(-percent * trader.getShares());
				return true;
			} else {
				 System.out.println("Insufficient number of shares to sell " + trade + " " +
				 time);
			}
		}
		return false;
	}

	public static void printTrades(List<Trade> tradeList) {

		System.out.println("\n ---------------- Trades ----------------- \n");
		for (int i = 0; i < tradeList.size(); ++i) {
			System.out.println(tradeList.get(i));
		}
		System.out.println(" ------------------------------------------");
	}

	public static void printInfoAboutTimeSegment(double[] input) {
		System.out.println("Standard Deviation " + Strategy.calculateStrdDev(input, Strategy.calculateMean(input)));
		System.out.println("Slope " + Strategy.calculateSlopeOfBestFit(input));
	}
}
