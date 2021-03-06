
import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.imgproc.Moments;
import org.opencv.ml.CvSVM;

import java.util.List;


public class Recognizer {

    // settings for seperating persons int the image
    static int paddHeightMax = 100;
    static int paddWidthMax = 110;
    static final int medianSize = 5; // need to be an odd number
    static final int minShapeSize = 300;

    // size of input image
    int imgHeight;
    int imgWidth;

    // person detection settings
    static final int detectBorder = 10;
    static final int rh = 10; // half size of rectangle around persons 

    // HoG
    static final int sizeCell = 6;
    static final int sizeBlock = 3;
    static final int hogBins = 9;
    static final int maxDeg = 360;
    
    static Mat hogFeatures;

    // SVM orientation
    CvSVM orientationSVM = new CvSVM();

    /*
    public Recognizer(Mat labels, Mat data) {
        orientationSVM = new CvSVM();
        orientationSVM.train(data, labels);
        //orientationSVM.save(modelFileName);
    }
    */

    /**
     * Loads a model from XML file.
     * @param filename 
     */
    public void loadModel(String filename) {
        orientationSVM.load(filename);
    }
    
    /**
     * Creates a model, then it stores to XML file in the same directory.
     * @param dir   path to directory with training data 
     */
    public void createModel(String dir) {
        Mat labels = new Mat(0, 0, CvType.CV_32FC1);
        Mat data = new Mat(0, 0, CvType.CV_32FC1);

        // get all file names
        List<String> listFileNames = getFilenames(dir);

        // create cells
        List<Rect> cells = genBlocks(paddHeightMax, paddWidthMax, sizeCell, sizeCell);

        // create blocks
        List<Rect> blocks = genBlocks((int) Math.floor(paddHeightMax / sizeCell), (int) Math.floor(paddWidthMax / sizeCell) * hogBins, sizeBlock, sizeBlock * hogBins);

        // compute feaures and get names of classes
        for (String f : listFileNames) {
            Mat img = Highgui.imread(dir+f, CvType.CV_8U);
            Mat hogFeature = hog(img, cells, blocks);
            float label = Float.parseFloat(f.substring(0, 1));
            
            Mat l = new Mat(1, 1, CvType.CV_32FC1, new Scalar(label));
            labels.push_back(l);
            
            data.push_back(hogFeature);
        }
        
        // train model
        orientationSVM.train(data, labels);
        
        // save model
        orientationSVM.save(dir + "model.xml");
    }

    /**
     * Predicts a class of given image according to created model.
     * @param sample    image
     * @return          predicted class
     */
    public float predict(Mat sample) {
        return orientationSVM.predict(sample);
    }

    /**
     * Computes histogram.
     * @param img       image from which is histogram computed
     * @param nbins     number of bins used for histogram computation
     * @param maxValue  maximum intensity value which will be examined
     * @return          1D histogram
     */
    private Mat compHist(Mat img, int nbins, int maxValue) {
        List<Mat> imgHist = Arrays.asList(img);
        MatOfInt histSize = new MatOfInt(nbins);
        MatOfFloat ranges = new MatOfFloat(0, maxValue);
        Mat hist = new Mat();

        Imgproc.calcHist(imgHist, new MatOfInt(0), new Mat(), hist, histSize, ranges);

        return hist;
    }

    /**
     * Creates a bin used for histogram later.
     * @param nbins     number of bins used for histogram computation
     * @param maxValue  maximum intensity value which will be examined
     * @return          bins
     */
    private List<Integer> createBins(int nbins, int maxValue) {
        int stepHist = Math.round(maxValue / nbins);
        List<Integer> bins = new ArrayList<>();

        for (int i = maxValue / nbins / 2; i < maxValue; i += stepHist) {
            bins.add(i);
        }

        return bins;
    }

    /**
     * Get all PNG file names from given directory.
     * @param dir   path to directory
     * @return      list of all PNG file names
     */
    public List<String> getFilenames(String dir) {
        List<String> results = new ArrayList<>();

        File[] files = new File(dir).listFiles();

        for (File file : files) {
            if (file.isFile() && file.getName().toLowerCase().endsWith(".png")) {
                results.add(file.getName());
            }
        }

        return results;
    }

    /**
     * Adds a padding to examined image to happen to be all images of the same size.
     * @param img   image
     * @return      image with padding
     */
    private Mat addPadding(Mat img) {
        Mat paddImg = new Mat();

        int extraVertPadd = 0;
        int extraHoriPadd = 0;

        int vert = (paddHeightMax - img.height()) / 2;
        int hori = (paddWidthMax - img.width()) / 2;

        Scalar black = new Scalar(0, 0, 0);

        Core.copyMakeBorder(img, paddImg, vert, vert, hori, hori, Core.BORDER_CONSTANT, black);

        if (paddImg.height() < paddHeightMax) {
            extraVertPadd = 1;
        }

        if (paddImg.width() < paddWidthMax) {
            extraHoriPadd = 1;
        }

        Core.copyMakeBorder(paddImg, paddImg, 0, extraVertPadd, 0, extraHoriPadd, Core.BORDER_CONSTANT, black);

        return paddImg;
    }

    /**
     * Preprocessing step of human detection.
     * @param img       image
     * @param mask      mask of scene
     * @param nbins     number of bins used for histogram computation
     * @param maxValue  maximum intensity value which will be examined
     * @return          filtered image
     */
    public Mat maskFilter(Mat img, Mat mask, int nbins, int maxValue) {
        // filtering background
        Mat diff = new Mat();
        Core.subtract(img, mask, diff);

        // compute histogram
        Mat hist = compHist(img, nbins, maxValue);

        // find how many elements at the end of histogram are zero valued
        int zPos = -1;
        for (int i = hist.rows() - 1; i >= 0; i--) {
            if (hist.get(i, 0)[0] != 0.0) {
                zPos = i + 1;
                break;
            }
        }

        // creating bins for histogram similar to GNU Octave ones
        List<Integer> h = createBins(nbins, maxValue);
        h = createBins(nbins, h.get(zPos));

        // finds first local minimum
        double tmpLast = hist.get(0, 0)[0];
        double tmp;
        int locMin = -1;

        for (int i = 1; i < hist.rows(); i++) {
            tmp = hist.get(i, 0)[0];

            if (tmp > tmpLast) {
                locMin = i;
                break;
            } else {
                tmpLast = tmp;
            }
        }

        // using final mask to filter background
        Mat binMask = new Mat();
        Imgproc.threshold(diff, binMask, h.get(locMin), 1, 0);

        return img.mul(binMask);
    }

    /**
     * Detects head from given image.
     * @param img       image
     * @param nbins     number of bins used for histogram computation
     * @param maxValue  maximum intensity value which will be examined
     * @return          position of head center
     */
    public Point detectHead(Mat img, int nbins, int maxValue) {
        int posBin = 2; // counted from the end

        // compute histogram
        Mat hist = compHist(img, nbins, maxValue);

        int newBorder = -1;

        for (int i = nbins - 1; i >= 0; i--) {
            if (hist.get(i, 0)[0] == 0) {
                newBorder = i;
            } else {
                break;
            }
        }

        // creating bins for histogram similar to GNU Octave ones
        List<Integer> h = createBins(nbins, maxValue);
        h = createBins(nbins, h.get(newBorder));

        // get only top of head
        Mat binMask = new Mat();
        int threshold = h.get(nbins - posBin);
        Imgproc.threshold(img, binMask, threshold, maxValue, 0);

        // find moments
        Moments mmnt;
        mmnt = Imgproc.moments(binMask);

        // get centroid
        Point p = new Point();
        p.x = mmnt.get_m10() / mmnt.get_m00();
        p.y = mmnt.get_m01() / mmnt.get_m00();

        return p;
    }

    /**
     * Creates a kernel.
     * @param dim   selection of kernel dimension 
     * @return      kernel
     */
    private Mat createKernel(boolean dim) {
        Mat kernel;

        if (dim) {
            kernel = new Mat(1, 3, CvType.CV_8S) {
                {
                    put(0, 0, -1);
                    put(0, 1, 0);
                    put(0, 2, 1);
                }
            };
        } else {
            kernel = new Mat(3, 1, CvType.CV_8S) {
                {
                    put(0, 0, -1);
                    put(1, 0, 0);
                    put(2, 0, 1);
                }
            };
        }

        return kernel;
    }

    // not ready for overlaping
    public List<Rect> genBlocks(int height, int width, int stepHeight, int stepWidth) {
        List<Rect> rects = new ArrayList<>();
        Rect tmpRect;

        for (int i = 0; i < width; i += stepWidth) {
            for (int j = 0; j < height; j += stepHeight) {
                tmpRect = new Rect(i, j, stepWidth, stepHeight);

                // skips incomplete blocks
                // these blocks does not contain any neccesary values
                if ((i + stepWidth) < width && (j + stepHeight) < height) {
                    rects.add(tmpRect);
                }
            }
        }

        return rects;
    }

    /**
     * Normalize values of matrix according to L2-norm.
     * @param mat   matrix
     * @return      normalized matrix
     */
    private Mat L2norm(Mat mat) {
        Mat nMat = new Mat();

        double norm = Core.norm(mat, 2);
        double squareNorm = Math.pow(norm, 2);
        double squareE = Math.pow(0.01, 2);

        double divisor = squareNorm + squareE;
        divisor = Math.sqrt(divisor);

        Mat divisorMat = new Mat();
        Scalar s = new Scalar(divisor);
        Core.multiply(Mat.ones(mat.size(), CvType.CV_32F), s, divisorMat);

        Core.divide(mat, divisorMat, nMat);
        return nMat;
    }

    /**
     * Computes histogram of oriented histograms from given image.
     * @param img       image
     * @param cells     coordinates of cells
     * @param blocks    coordinates of blocks
     * @return          features 
     */
    public Mat hog(Mat img, List<Rect> cells, List<Rect> blocks) {
        Mat features = new Mat();

        img.convertTo(img, CvType.CV_32F);

        // create convolutional kernels
        Mat kernely = createKernel(Boolean.FALSE);
        Mat kernelx = createKernel(Boolean.TRUE);

        // compute gradients
        Mat dy = new Mat(img.size(), CvType.CV_32F);
        Mat dx = new Mat(img.size(), CvType.CV_32F);

        Imgproc.filter2D(img, dy, -1, kernely);
        Imgproc.filter2D(img, dx, -1, kernelx);

        // compute magnitudes
        // r = sqrt(dx.^2 + dy.^2);
        Mat hypx = new Mat(img.size(), CvType.CV_32F);
        Mat hypy = new Mat(img.size(), CvType.CV_32F);
        Mat tmpMat = new Mat(img.size(), CvType.CV_32F);
        Mat mag = new Mat(img.size(), CvType.CV_32F);
        Mat r = new Mat(img.size(), CvType.CV_32F);

        Core.pow(dx, 2, hypx);
        Core.pow(dy, 2, hypy);

        Core.add(hypx, hypy, tmpMat);

        Core.sqrt(tmpMat, mag);

        // TODO: check if it is really need?
        // used in Octave to avoid negative values
        //Core.absdiff(dx, Mat.zeros(img.size(), CvType.CV_8U), dx);
        Core.magnitude(dx, dy, r);
        //Mat sub;
        Mat vectHist = new Mat();
        List<Mat> listHist = new ArrayList<>();

        // compute histogram from particular cells
        for (Rect rect : cells) {
            Mat sub = img.submat(rect);

            // TODO: check the correctnes of histogram computation
            listHist.add(compHist(sub, hogBins, maxDeg));
        }

        // horizontal concatenation of all histograms
        Core.vconcat(listHist, vectHist);

        // reshape vector to matrix
        vectHist = vectHist.reshape(0, (int) Math.floor(paddHeightMax / sizeCell));

        List<Mat> listNorm = new ArrayList<>();
        Mat L2normTmp;

        // normalize histograms in particular blocks
        for (Rect rect : blocks) {
            Mat sub = vectHist.submat(rect);
            L2normTmp = L2norm(sub);
            L2normTmp = L2normTmp.reshape(0, 1);
            listNorm.add(L2normTmp);
        }

        Core.hconcat(listNorm, features);

        return features;
    }
    
    /**
     * Detects a human body in image and computes features from it.
     * @param img       image
     * @param nbins     number of bins used for histogram computation
     * @param maxValue  maximum intensity value which will be examined
     */
    public void detect(Mat img, int nbins, int maxValue) {
        // filtering
        Mat tmpImg = new Mat();
        Imgproc.medianBlur(img, tmpImg, medianSize);

        Mat binMask = new Mat();
        Imgproc.threshold(tmpImg, binMask, 1, 1, 0);

        ArrayList<MatOfPoint> contours = new ArrayList<>();
        Mat hierarchy = new Mat();
        Imgproc.findContours(binMask, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        // create cells
        List<Rect> cells = genBlocks(paddHeightMax, paddWidthMax, sizeCell, sizeCell);

        // create blocks
        List<Rect> blocks = genBlocks((int) Math.floor(paddHeightMax / sizeCell), (int) Math.floor(paddWidthMax / sizeCell) * hogBins, sizeBlock, sizeBlock * hogBins);

        for (MatOfPoint i : contours) {
            if (Imgproc.contourArea(i) > minShapeSize) {
                Rect rect = Imgproc.boundingRect(i);
                Mat sub = img.submat(rect);

                // add padding to image
                Mat paddImg = addPadding(sub);
                hogFeatures = hog(paddImg, cells, blocks);
            }
        }
    }
}