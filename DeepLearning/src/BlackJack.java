import java.util.ArrayList;


public class BlackJack {

	public void createDeck() { 
		int deckSize = 312;
		ArrayList<Integer> deck = new ArrayList<Integer>();  
			for(int i=0; i<deckSize; i++) {
				deck.add((int) ((Math.random() *311) + 1));
			}
	
	}
}
