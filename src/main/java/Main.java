

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Rect;
import org.opencv.highgui.Highgui;

import java.io.IOException;
import java.util.List;

public class Main {

    public static void main(String[] args) throws IOException {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);

        String imgFilename = "src/main/resources/Jackie-pass.png";
        String maskFilename = "src/main/resources/Jackie-selfie.jpg";

        Mat mask = Highgui.imread(maskFilename);
        Mat img = Highgui.imread(imgFilename);
        
//        //DataHandler dataHandler = new DataHandler();
//        Mat data = dataHandler.readCsv("data.csv");
//        Mat trainLabels = dataHandler.getLabel(data);
//        Mat trainData = dataHandler.getData(data);
//        Recognizer recognizer = new Recognizer(trainLabels, trainData);
                
        // ORIENTATION
        Recognizer orientation = new Recognizer();
        orientation.imgHeight = img.rows();
        orientation.imgWidth = img.cols();
        
        // LOAD MODEL
        String modelFileName = "model.xml";
        orientation.loadModel(modelFileName);
        
        // CREATE AND SAVE MODEL
        //orientation.createModel("/path/to/training/images/");

        // settings
        int maxValue = 255; // max intensity value of grayscale image
        int nbins = 20; // number of bins
        
        // IMAGE PREPROCESSING
        Mat filt = orientation.maskFilter(img, mask, nbins, maxValue);
        orientation.detect(filt, nbins, maxValue);
        
        // USE MODEL
        float direction = orientation.predict(orientation.hogFeatures);
        System.out.println(direction);
        
        ////////////////////////////////////////////////////////////////////////
        
        imgFilename = "image.png";
        Mat img2 = Highgui.imread(imgFilename, CvType.CV_8U); 
        
        // GENDER
        Recognizer gender = new Recognizer();
        int imgSize = 71;
        gender.paddHeightMax = imgSize;
        gender.paddWidthMax = imgSize;
        
        // CREATE AND SAVE MODEL
        gender.createModel("/path/to/training/images/");
        
        int cellSize = 6;
        int blockSize = 3;
        nbins = 9;
        
        // create cells
        List<Rect> cells = gender.genBlocks(imgSize, imgSize, cellSize, cellSize);

        // create blocks
        List<Rect> blocks = gender.genBlocks((int) Math.floor(imgSize / cellSize), (int) Math.floor(imgSize / cellSize)  *  nbins, blockSize, blockSize * nbins);
        
        // USE MODEL
        float g = gender.predict(gender.hog(img2, cells, blocks));
        System.out.println(g);
        
        ////////////////////////////////////////////////////////////////////////
        

    }
}