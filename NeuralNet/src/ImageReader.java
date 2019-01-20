import java.awt.Color;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import javax.imageio.ImageIO;

public class ImageReader {

	public List<double[][][]> readImageFile(String fileName) {
		List<double[][][]> imageCollection = null; 
		try{
			File path = new File(fileName);
			File[] files = path.listFiles();
			imageCollection = new ArrayList<double[][][]>();
			
			for (int i = 0; i < files.length; i++){
			    if (files[i].isFile()){
			        imageCollection.add(readImage(files[i]));
			    }
			}
			
		} catch (Exception e) {
			System.err.println(e.getMessage());
		}
		
		return imageCollection; 
	}

	public double[][][] readImage(File file) {
		BufferedImage image = null; 
		try {
			image = ImageIO.read(file); 
		} catch (IOException e) {
			System.err.println(e.getMessage());
		}
		return parseImageToRGB(image);  //may need error handling for null value
	
	}
	
	public double[][][] parseImageToRGB(BufferedImage image) {
		int w = image.getWidth();
	    int h = image.getHeight();
	    
	    
	    double[][][] rgbImage = new double[3][h][w]; 
	    int[] colors = new int[w*h];
	    
		image.getRGB(0, 0, image.getWidth(), image.getHeight(), colors, 0, image.getWidth()); 
		
		 
		 int[] red = new int[colors.length];
		 int[] green = new int[colors.length];
		 int[] blue = new int[colors.length];
		 
		 
		 
		 for (int i = 0; i < colors.length; i++) {
		      Color color = new Color(colors[i]);
		      red[i] = color.getRed();
		      green[i] = color.getGreen();
		      blue[i] = color.getBlue();
		 }
		 
		 int counter = 0;
		 
		 for(int i=0; i<3; i++) {
			 for(int j=0; j<h; j++) {
				 for(int k=0; k<w; k++) {
					 if(i==0) {
						 rgbImage[i][j][k] = red[counter]; 
					 } else if (i == 1) {
						 rgbImage[i][j][k] = blue[counter];
					 } else if(i == 2) {
						 rgbImage[i][j][k] = green[counter]; 
					 }
					 counter++; 
				 }
			 }
			 counter = 0;
		 }
	    
	    System.out.println(w);
	    System.out.println(h);
	    
	    return rgbImage; 
	}
}
