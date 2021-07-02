import java.util.ArrayList;
import java.util.List;

public class Strategy {

	int startTime; // where to start using the strategy / where to start looking for the strategy
	int endTime; // where to stop using the strategy

	public Strategy(String strategy) { // assumes full use of market data
		startTime = 1;
	}

	public Strategy(int startTime, int endTime, String strategy) {
		this.startTime = startTime;
		this.endTime = endTime;
	}

	public Strategy(int startTime, String strategy) {
		this.startTime = startTime;
	}

	// cannot allow access to marketInfo past time to simulate current time
	public Trade basicStrategy(double[] marketInfo, int time) { // time refers to how far ahead the program is allowed
																// to look into the array
		if (time >= 3) {
			if (marketInfo[time] >= 1.02 * marketInfo[time - 3]) {
				return new Trade(marketInfo[time], time, 1 * (marketInfo[time] / marketInfo[time - 3]), "SELL",
						"VALUE"); // optimize constructor, change to time based
			} else if (marketInfo[time] <= .98 * marketInfo[time - 3]) {
				return new Trade(marketInfo[time], time, 1 * (marketInfo[time - 3] / marketInfo[time]), "BUY", "VALUE"); // optimize
																															// constructor,
																															// change
																															// to
																															// time
																															// based
			}
		}
		return new Trade("NULL");
	}

	public Trade stabilityTrading(double[] marketInfo, List<Trade> tradeList, int time) {
		if (time >= 5) {
			double[] segment = getArraySegment(marketInfo, time - 5, time);

			// System.out.println(calculateStrdDev(segment, calculateMean(segment)));
			// System.out.println(calculateSlopeOfBestFit(segment));

			if (calculateSlopeOfBestFit(segment) <= -200) { // buy condition
				return new Trade(marketInfo[time], time, 1 * (marketInfo[time - 5] / marketInfo[time]), "BUY", "VALUE");
			} else if (calculateStrdDev(segment, calculateMean(segment)) <= 300) { // sell condition
				return new Trade(marketInfo[time], time, 1 * (marketInfo[time] / marketInfo[time - 5]), "SELL",
						"VALUE");
			}
		}
		return new Trade("NULL");
	}

	// global slopeList
	static List<Double> slopeList = new ArrayList<Double>();

	public Trade slopeTrading(Trader trader, double[] marketInfo, int time, int sampleSize) {
		double fiftyDayAverage = 1;
		double overUnderValueAdjustment = 1;

		if (time >= sampleSize) { // time condition

			if (time >= 50) {
				fiftyDayAverage = calculateMean(getArraySegment(marketInfo, time - 50, time));

				if (marketInfo[time] > fiftyDayAverage) { // overvalued
					overUnderValueAdjustment = (marketInfo[time] / fiftyDayAverage);
				} else { // undervalued
					overUnderValueAdjustment = (fiftyDayAverage / marketInfo[time]);
				}
				// System.out.println("OVERUNDER " + overUnderValueAdjustment);
			}

			if (slopeList.isEmpty() || ((slopeList.get(slopeList.size() - 1)
					* calculateSlopeOfBestFit(getArraySegment(marketInfo, time - sampleSize, time))) > 0)) { // Multiplication
				// used to check
				// signs, may be
				// slow
				slopeList.add(calculateSlopeOfBestFit(getArraySegment(marketInfo, time - sampleSize, time)));
			} else { // run alg for a buy or sell
				double ram = RAM(slopeList) / marketInfo[time];

				slopeList.clear();
				slopeList.add(calculateSlopeOfBestFit(getArraySegment(marketInfo, time - sampleSize, time)));
				if (ram < -0.02) { // buy condition
					return new Trade(1 * marketInfo[time], time, 1, "BUY", "VALUE");

				} else if (ram > 0.05) { // sell condition
					return new Trade(1 * marketInfo[time], time, 1, "SELL", "VALUE");
				}
			}
		}

		// System.out.println("Return null");
		return new Trade("NULL");
	}

	public Trade percentTrading(Trader trader, double[] marketInfo, int time, int sampleSize) { // trades are based on
																								// percent swings
		if (time >= sampleSize) {
			if (marketInfo[time] < .96 * marketInfo[time - sampleSize]) {
				return new Trade(1 * marketInfo[time], time, 5, "BUY", "TIME");
			} else if (marketInfo[time] > 1.04 * marketInfo[time - sampleSize]) {
				return new Trade(1 * marketInfo[time], time, 5, "SELL", "TIME");
			}
		}

		return new Trade("NULL");
	}

	public Trade percentAverageTrading(Trader trader, double[] marketInfo, int time, int sampleSize) { // trades are
																										// based on
																										// percent
																										// swings
		if (time >= sampleSize) {

			double fiftyDayAverage = 1;

			if (time >= 10) {
				fiftyDayAverage = calculateMean(getArraySegment(marketInfo, time - 10, time));
				// System.out.println("OVERUNDER " + overUnderValueAdjustment);
			}

			if (marketInfo[time] < .97 * fiftyDayAverage) {
				return new Trade(1 * marketInfo[time], time, 5, "BUY", "TIME");
			} else if (marketInfo[time] > 1.10 * fiftyDayAverage) {
				return new Trade(1 * marketInfo[time], time, 5, "SELL", "TIME");
			}
		}

		return new Trade("NULL");
	}

	public Trade percentAverageTradingLoop(Trader trader, double[] marketInfo, int time, int var1, double var2,
			double var3, double var4, double var5) { // trades are based on percent swings

		double fiftyDayAverage = 1;

		if (time >= var1) {
			fiftyDayAverage = calculateMean(getArraySegment(marketInfo, time - var1, time));
			// System.out.println("OVERUNDER " + overUnderValueAdjustment);

			if (marketInfo[time] < var2 * fiftyDayAverage) {
				return new Trade(1 * marketInfo[time], time, 5, "BUY", "TIME");
			} else if (marketInfo[time] > var3 * fiftyDayAverage) {
				return new Trade(1 * marketInfo[time], time, 5, "SELL", "TIME");
			}
		}

		return new Trade("NULL");
	}

	public Trade slopeTradingLoop(Trader trader, double[] marketInfo, int time, int sampleSize, double buyAmount,
			double sellAmount, double buyCostMultiplier, double sellCostMultiplier, double buyCondition,
			double sellCondition) {
		double fiftyDayAverage = 1;
		double overUnderValueAdjustment = 1;

		if (time >= sampleSize) { // time condition

			if (time >= 50) {
				fiftyDayAverage = calculateMean(getArraySegment(marketInfo, time - 50, time));

				// if(marketInfo[time] > fiftyDayAverage) { //overvalued
				// overUnderValueAdjustment = (marketInfo[time] - fiftyDayAverage) /
				// marketInfo[time];
				overUnderValueAdjustment = (marketInfo[time] / fiftyDayAverage);
				// }
				// } else { //undervalued
				// overUnderValueAdjustment = (fiftyDayAverage - marketInfo[time]) /
				// fiftyDayAverage;
				// overUnderValueAdjustment = (fiftyDayAverage / marketInfo[time]);
				// }
				// System.out.println("OVERUNDER " + overUnderValueAdjustment);
			}

			// overUnderValueAdjustment = 1;

			// double slope = calculateSlopeOfBestFit(getArraySegment(marketInfo, time -
			// sampleSize, time));

			// if (slopeList.isEmpty() || ((slopeList.get(slopeList.size() - 1)
			// * slope) > 0)) { // Multiplication
			// used to check
			// signs, may be
			// slow

			// System.out.println(overUnderValueAdjustment);
			// System.out.println("ADDED");
			// slopeList.add(slope);
			// } else { // run alg for a buy or sell
			// double ram = RAM(slopeList) / marketInfo[time];

			// slopeList.clear();
			// System.out.println("RAM " + ram + " at time " + ((time-8)*100 + 1000));
			// slopeList.add(slope);
			if (overUnderValueAdjustment < buyCondition) { // buy condition
				// System.out.println("BUY COND");
				// System.out.println(time + " " + marketInfo[time] + " " +
				// calculateSlopeOfBestFit(getArraySegment(marketInfo, time - sampleSize,
				// time)));
				return new Trade(buyCostMultiplier * marketInfo[time], time, buyAmount, "BUY", "VALUE");
				// return new Trade(buyCostMultiplier*marketInfo[time], time,
				// overUnderValueAdjustment*buyAmount, "BUY", "TIME");
				// return new Trade(marketInfo[time], time, -ram/1000, "BUY", "TIME");
			} else if (overUnderValueAdjustment > sellCondition) {
				// System.out.println("SELL COND");
				return new Trade(sellCostMultiplier * marketInfo[time], time, sellAmount, "SELL", "VALUE");
				// return new Trade(sellCostMultiplier*marketInfo[time], time,
				// overUnderValueAdjustment*sellAmount, "SELL", "TIME");
				// return new Trade(marketInfo[time], time, ram/1000, "SELL", "TIME");
			}
		}
		// }

		// System.out.println("Return null");
		return new Trade("NULL");
	}

	public static double calculateStrdDev(double[] inputs, double mean) {
		double strdDev = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			strdDev += Math.pow(inputs[i] - mean, 2); // changed
		}

		strdDev /= inputs.length;

		return Math.sqrt(strdDev);
	}

	public static double calculateMean(double[] inputs) {
		double runningTotal = 0.0;

		for (int i = 0; i < inputs.length; i++) {
			runningTotal += inputs[i]; // location of the closing price
		}

		return runningTotal / inputs.length;
	}

	public static double calculateSlopeOfBestFit(double[] inputs) { // built for equally spaced intervals
		double mean = calculateMean(inputs);
		double meanOfX = (inputs.length + 1) / 2.0;
		double numerator = 0;
		double denominator = 0;

		for (int i = 1; i <= inputs.length; i++) { // makes calculation above easier
			numerator += ((inputs[i - 1] - mean) * (i - meanOfX));
			// }

			// for (int i = 1; i <= inputs.length; i++) { // probably a mathematical formula
			// for this
			denominator += ((i - meanOfX) * (i - meanOfX));
		}
		return numerator / denominator;
	}

	public static double[] getArraySegment(double[] input, int start, int end) {
		double[] newSegment = new double[end - start];
		for (int i = 0; i < (end - start); ++i) {
			newSegment[i] = input[i + start];
		}
		return newSegment;
	}

	public static double RAM(List<Double> slopeList) {
		double total = 0.0;

		for (int i = 1; i < slopeList.size() - 1; ++i) { // left ram, assumes equal spacing between points
			total += 2 * slopeList.get(i);
		}

		total += (slopeList.get(0) + slopeList.get(slopeList.size() - 1));

		return (total / 2); // averages rams, can be made more efficient with use of endpoints and doubling
							// midpoints
	}

}
