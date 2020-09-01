import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

/**
 * Class ModelSaver is, as the name implies, used to save a network model once it has been trained
 * After the training is finished the method save model is called 
 *
 */
public class ModelSaver {
	
	/**
	 * saveModel runs all the functions within this class in the right order to save a model 
	 * @param model: the network model to be saved
	 * @param weights  //check if needed TODO 
	 */
	public void saveModel(NetworkModel model, Weights weights) {
		promptUser();
		createFolder();
		saveFileToFolder();
		saveModelToFile(model, weights);
	}

	String input = ""; //input from the user 
	String fileName = ""; //name of the file to write to
	Scanner scan = new Scanner(System.in);

	/**
	 * promptUser() asks the user if they want to name the file and what they would name it if they do 
	 * If they don't want to save the model the program is terminated
	 */
	private void promptUser() {
		System.out.println();
		System.out.println("Would you like to save the model?");
		input = scan.nextLine();

		if (input.equalsIgnoreCase("y") || input.equalsIgnoreCase("yes")) {
			System.out.println("What would you like to name the file?");
			fileName = scan.nextLine();
		} else {
			System.out.println("Terminating program");
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			System.exit(0);
		}

	}

	String filePath = System.getProperty("user.home") + "\\Desktop\\Models";

	/**
	 * Checks if a "Models" folder exists in the fileSystem at the location of filePath
	 * If the folder doesn't exist then the folder is created at that location 
	 */
	private void createFolder() {
		Path path = Paths.get(filePath); //path objects hold a string object of a file location 
		if (!Files.exists(path)) {
			System.out.println("Folder created");
			try {
				Files.createDirectories(path); //creates all necessary directories mentioned in path
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}

	/**
	 * Saves an empty textFile in the models folder (see above) 
	 * While a file exists in the models file with the name "fileName" the program will continue to ask you for a unique name for the file
	 * Then once a unique fileName is found, an empty text file is created
	 */
	private void saveFileToFolder() {
		Path path = Paths.get(filePath + "\\" + fileName + ".txt");

		while (Files.exists(path)) {
			path = Paths.get(filePath + "\\" + fileName + ".txt");
			if (Files.exists(path)) {
				System.out.println("A file with that name already exists, please choose another");
				fileName = scan.nextLine();
			}
		}

	}

	/**
	 * saveModel to file takes all the necessary information from the network to reproduce the results and writes them to the file "fileName" 
	 * in the models folder
	 * 
	 * Using a buffered output stream, information is printed into a text document that gets read by the network evaluator for testing 
	 * 
	 * 
	 * @param model
	 * @param weights
	 */
	private void saveModelToFile(NetworkModel model, Weights weights) {
		int counter = 1; //counts the number of times something is printed in a large list to keep the formatting more of less easy to read
		String space = " ";
		List<double[][]> weightList = weights.weightList; 

		List<Layer> list = model.layerList;
		InputLayer inputLayer =  (InputLayer) model.layerList.get(0); 
	
		try {

			BufferedOutputStream bos = new BufferedOutputStream(
					new FileOutputStream(filePath + "\\" + fileName + ".txt"));

			bos.write((list.size() + "").getBytes()); //Print number of layers in the network 

			for (int i = 0; i < list.size(); i++) {
				bos.write((space + "").getBytes());
				bos.write((Layer.parseObjectTypeIntoInt(list.get(i)) + "").getBytes()); //for each layer print the type of layer it is 
			}

				for (int i = 0; i < list.size(); i++) { // layer sizes
					bos.write((space + "").getBytes());
					bos.write((list.get(i).layerSize + "").getBytes()); //for each layer print the size of each layer
				}

				bos.write(System.lineSeparator().getBytes());

				for (int i = 1; i < list.size(); i++) {
					bos.write((Activator.convertActivationString(list.get(i).activation) + "").getBytes()); //for each layer print the activation function associated with each layer
					bos.write((space + "").getBytes());
				}

				bos.write(System.lineSeparator().getBytes());

				for (int i = 0; i < weightList.get(0).length - 1; i++) { //TODO check if can change to the num of input (size of mean array should relate directly
																		//to the number of columns) (still need to check if the bias column is included in the size) 
					bos.write((inputLayer.normalizer.meanArray[i] + "").getBytes()); //for each column of the inputLayer print the mean for the column
					bos.write(System.lineSeparator().getBytes()); //
				}
				for (int i = 0; i < weightList.get(0).length - 1; i++) {
					bos.write((inputLayer.normalizer.strdDev[i] + "").getBytes()); //for each column of the inputLayer print the standard deviation for the column 
					bos.write(System.lineSeparator().getBytes());
				}

				for (int i = 0; i < weightList.size(); i++) {
					for (int j = 0; j < weightList.get(i).length; j++) {
						for (int k = 0; k < weightList.get(i)[0].length; k++) {
							bos.write((weightList.get(i)[j][k] + "").getBytes()); //sequentially print all the weights used in the network after training
							bos.write((space + "").getBytes());
							if (counter % 30 == 0) { //after 30 weights have been printed add a new line to the text document
								bos.write(System.lineSeparator().getBytes());
							}
							counter++;
						}
					}
				}

				//TODO is this even necessary 
				for (int i = 1; i < list.size(); i++) {

					//TODO: major potential error in printing out the layerSizes and activations of all the layers twice
						bos.write((space + "").getBytes());
						bos.write((list.get(i).layerSize + "").getBytes()); //for each layer after the input layer print the size of the layers
						
						bos.write((space + "").getBytes());
						bos.write((Activator.convertActivationString(list.get(i).activation) + "").getBytes());					
				
						//made larger change here, test if it works 
				}
				
				//TODO: ACCIDENTALLY DUPLICATED CODE!!!???? : HAS BEEN DELETED

			bos.close();
		} catch (IOException e) {
			System.out.println("Exception " + e);
		}
	}

} // end of class
