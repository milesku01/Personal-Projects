import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;


public class ModelSaver {
	
	public void saveModel(NetworkModel model, Weights weights) {
		promptUser(); 
		createFolder();
		saveFileToFolder();
		saveModelToFile(model, weights);
	}

	String input = ""; 
	String fileName = ""; 
	Scanner scan = new Scanner(System.in);
	
	private void promptUser()  {
		System.out.println();
		System.out.println("Would you like to save the model?");
		input = scan.nextLine(); 
		
		if (input.equalsIgnoreCase("y") || input.equalsIgnoreCase("yes")){
			System.out.println("What would you like to name the file?");
			fileName = scan.nextLine(); 
		} else {
			System.out.println("Terminating program");
			try {
				Thread.sleep(1000);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} 
			System.exit(0);
		}
		
	}
	
	String filePath = System.getProperty("user.home") + "\\Desktop\\Models";
	private void createFolder() {
		Path path = Paths.get(filePath);
		if (!Files.exists(path)) {
			System.out.println("Folder created");
			try {
				Files.createDirectories(path);
			} catch (IOException e) {
				e.printStackTrace();
			}
		} 
	}
	
	private void saveFileToFolder()  {
		Path path = Paths.get(filePath + "\\"  + fileName + ".txt");	
		
		while(Files.exists(path)) {
			path = Paths.get(filePath + "\\"  + fileName + ".txt");
			System.out.println("A file with that name already exists, please choose another");
			fileName = scan.nextLine(); 
		}

	}
	
	private void saveModelToFile(NetworkModel model, Weights weights) {
		int counter = 1; 
		String space = " "; 
		List<Layer> list = model.layerList; 
		InputLayer inputLayer = (InputLayer) model.layerList.get(0); //check if reference problem
		
		try {
			
		BufferedOutputStream bos = new BufferedOutputStream(new FileOutputStream(filePath + "\\"  + fileName + ".txt")); 
	
		for(int i=0; i<list.size(); i++) {  //layer sizes
			bos.write((list.get(i).layerSize + "").getBytes()); 
			bos.write((space + "").getBytes());
		}
		
		bos.write(System.lineSeparator().getBytes());
		
		for(int i =0; i<weights.weightList.get(0).length-1; i++) {
			bos.write((inputLayer.normalizer.meanArray[i] + "").getBytes());
			bos.write(System.lineSeparator().getBytes());
		} 
		for(int i=0; i<weights.weightList.get(0).length-1; i++){
			bos.write((inputLayer.normalizer.strdDev[i] + "").getBytes());
			bos.write(System.lineSeparator().getBytes());
		}
		
		for(int i=0; i<weights.weightList.size(); i++) {
			for(int j=0; j<weights.weightList.get(i).length; j++) {
				for(int k=0; k<weights.weightList.get(i)[0].length; k++) {
					bos.write((weights.weightList.get(i)[j][k] + "").getBytes());
					bos.write((space + "").getBytes());
					if(counter%30 == 0) {
						bos.write(System.lineSeparator().getBytes());
					} 
					counter++;
				}
			}
		}
		bos.close();
		} catch (IOException e){
			System.out.println("Exception " + e);
		}
	}
		
} //end of class
