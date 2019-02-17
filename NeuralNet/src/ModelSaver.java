import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;

public class ModelSaver {
	Layer layer = new Layer();
	Activator activator = new Activator();

	public void saveModel(NetworkModel model, Weights weights) {
		promptUser();
		createFolder();
		saveFileToFolder();
		saveModelToFile(model, weights);
	}

	String input = "";
	String fileName = "";
	Scanner scan = new Scanner(System.in);

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

	private void saveModelToFile(NetworkModel model, Weights weights) {
		int counter = 1;
		int activationInt;
		String space = " ";

		List<Layer> list = model.layerList;
		InputLayer inputLayer = null;

		if (model.layerList.get(0) instanceof InputLayer) {
			inputLayer = (InputLayer) model.layerList.get(0); // check if reference problem
		}

		try {

			BufferedOutputStream bos = new BufferedOutputStream(
					new FileOutputStream(filePath + "\\" + fileName + ".txt"));

			bos.write((list.size() + "").getBytes());

			for (int i = 0; i < list.size(); i++) {
				bos.write((space + "").getBytes());
				bos.write((layer.parseObjectTypeIntoInt(list.get(i)) + "").getBytes());
			}

			if (list.get(0) instanceof InputLayer) {

				for (int i = 0; i < list.size(); i++) { // layer sizes
					bos.write((space + "").getBytes());
					bos.write((list.get(i).layerSize + "").getBytes());
				}

				bos.write(System.lineSeparator().getBytes());

				for (int i = 1; i < list.size(); i++) {
					activationInt = activator.convertActivationString(list.get(i).activation);
					bos.write((activationInt + "").getBytes());
					bos.write((space + "").getBytes());
				}

				bos.write(System.lineSeparator().getBytes());

				for (int i = 0; i < weights.weightList.get(0).length - 1; i++) {
					bos.write((inputLayer.normalizer.meanArray[i] + "").getBytes());
					bos.write(System.lineSeparator().getBytes());
				}
				for (int i = 0; i < weights.weightList.get(0).length - 1; i++) {
					bos.write((inputLayer.normalizer.strdDev[i] + "").getBytes());
					bos.write(System.lineSeparator().getBytes());
				}

				for (int i = 0; i < weights.weightList.size(); i++) {
					for (int j = 0; j < weights.weightList.get(i).length; j++) {
						for (int k = 0; k < weights.weightList.get(i)[0].length; k++) {
							bos.write((weights.weightList.get(i)[j][k] + "").getBytes());
							bos.write((space + "").getBytes());
							if (counter % 30 == 0) {
								bos.write(System.lineSeparator().getBytes());
							}
							counter++;
						}
					}
				}

			} else if (list.get(0) instanceof ConvolutionalLayer) {
				ConvolutionalLayer conv = (ConvolutionalLayer) list.get(0);

				bos.write((space + "").getBytes());
				bos.write((conv.normalizer.imageMean.length + "").getBytes());
			
				
				for(int i=0; i<conv.normalizer.imageMean.length; i++) {
					bos.write((space + "").getBytes());
					bos.write((conv.normalizer.imageMean[i] + "").getBytes());
				}	
				
				for(int i=0; i<conv.normalizer.imageMean.length; i++) {
					bos.write((space + "").getBytes());
					bos.write((conv.normalizer.imageStrdDev[i] + "").getBytes());
				}
				
				if (conv.type == "TEXT") {

					bos.write((space + "").getBytes());
					bos.write((conv.imageHeight + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.imageWidth + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.channelDepth + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.numofFilters + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.filterSize + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.strideLength + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((0 + "").getBytes()); // used for padding, because only zero type for now

				} else if (conv.type == "IMAGE") {

					bos.write((space + "").getBytes());
					bos.write((conv.numofFilters + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.filterSize + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((conv.strideLength + "").getBytes());

					bos.write((space + "").getBytes());
					bos.write((0 + "").getBytes()); // used for padding, because only zero type for now
				}

				for (int i = 1; i < list.size(); i++) {

					if (list.get(i) instanceof PoolingLayer) {
						PoolingLayer pool = (PoolingLayer) list.get(i);
						bos.write((space + "").getBytes());
						bos.write((pool.poolSize + "").getBytes());
					} else if (list.get(i) instanceof HiddenConvolutionalLayer) {
						HiddenConvolutionalLayer hidden = (HiddenConvolutionalLayer) list.get(i);
						bos.write((space + "").getBytes());
						bos.write((hidden.numofFilters + "").getBytes());

						bos.write((space + "").getBytes());
						bos.write((hidden.filterSize + "").getBytes());

						bos.write((space + "").getBytes());
						bos.write((hidden.strideLength + "").getBytes());

						bos.write((space + "").getBytes());
						bos.write((0 + "").getBytes()); // for padding

					} else if (list.get(i) instanceof HiddenLayer) {
						HiddenLayer hidden = (HiddenLayer) list.get(i);
						bos.write((space + "").getBytes());
						bos.write((hidden.layerSize + "").getBytes());

						activationInt = activator.convertActivationString(list.get(i).activation);

						bos.write((space + "").getBytes());
						bos.write((activationInt + "").getBytes());

					} else if (list.get(i) instanceof OutputLayer) {

						OutputLayer out = (OutputLayer) list.get(i);

						bos.write((space + "").getBytes());
						bos.write((out.layerSize + "").getBytes());

						activationInt = activator.convertActivationString(list.get(i).activation);

						bos.write((space + "").getBytes());
						bos.write((activationInt + "").getBytes());
					}

				}

				int hiddenCounter = 0;

				for (int i = 0; i < list.size(); i++) {
					if (list.get(i) instanceof ConvolutionalLayer) {
						hiddenCounter = 0;
						for (int j = 0; j < weights.filterList.get(0).threeDFilterArray.size(); j++) {
							for (int k = 0; k < weights.filterList.get(0).threeDFilterArray.get(j).length; k++) {
								for (int l = 0; l < weights.filterList.get(0).threeDFilterArray.get(j)[0].length; l++) {
									for (int m = 0; m < weights.filterList.get(0).threeDFilterArray
											.get(j)[0][0].length; m++) {
										bos.write((space + "").getBytes());
										bos.write((weights.filterList.get(0).threeDFilterArray.get(j)[k][l][m] + "")
												.getBytes());
										if (counter % 30 == 0) {
											bos.write(System.lineSeparator().getBytes());
										}
										counter++;
									}
								}
							}
						}
						bos.write(System.lineSeparator().getBytes());
						hiddenCounter++;

					} else if (list.get(i) instanceof HiddenConvolutionalLayer) {
						for (int j = 0; j < weights.filterList.get(hiddenCounter).twoDFilterArray.size(); j++) {
							for (int k = 0; k < weights.filterList.get(hiddenCounter).twoDFilterArray
									.get(j).length; k++) {
								for (int l = 0; l < weights.filterList.get(hiddenCounter).twoDFilterArray
										.get(j)[0].length; l++) {
									bos.write((space + "").getBytes());
									bos.write((weights.filterList.get(hiddenCounter).twoDFilterArray.get(j)[k][l] + "")
											.getBytes());
									if (counter % 30 == 0) {
										bos.write(System.lineSeparator().getBytes());
									}
									counter++;
								}
							}
						}
						bos.write(System.lineSeparator().getBytes());
						hiddenCounter++;
						
					} else if (list.get(i) instanceof HiddenLayer) {
						for (int l = 0; l < weights.weightList.size(); l++) {
							for (int j = 0; j < weights.weightList.get(l).length; j++) {
								for (int k = 0; k < weights.weightList.get(l)[0].length; k++) {
									bos.write((space + "").getBytes());
									bos.write((weights.weightList.get(l)[j][k] + "").getBytes());
									if (counter % 30 == 0) {
										bos.write(System.lineSeparator().getBytes());
									}
									counter++;
								}
							}
						}
					}
				}
			}

			bos.close();
		} catch (IOException e) {
			System.out.println("Exception " + e);
		}
	}

} // end of class
