import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;


public class Classifier {

    private static final int seed = 123;
    private static final int height = 64;
    private static final int width = 64;
    private static final int numExamples = 403;
    private static final int outputNum = 4;
    private static final int channels = 3;

    public static void main (String[] args) throws IOException, InterruptedException {

        // Initialize general data parameters

        int batchSize = 10;
        boolean appendLabels = true;
        int nFolds = 10;
        double splitTrainPerc = 1.0-(1.0/nFolds);

        // Define hyperparameters

        double[] l2Vals = new double[]{.0001,.0005, .001, .005};
        double[] dropVals = new double[]{0,.5, .9};
        MultiLayerConfiguration[] models = new MultiLayerConfiguration[l2Vals.length*dropVals.length];
        String[] modelParams = new String[l2Vals.length*dropVals.length];

        // Build models

        int index = 0;
        for(int i = 0; i < l2Vals.length; i++) {
            for (int j = 0; j < dropVals.length; j++) {
                models[index] = makeNetwork(l2Vals[i], dropVals[j]);
                modelParams[index] = new String("L2 = " + l2Vals[i] + ", dropout = " + dropVals[j]);
                index++;
            }
        }

        // File and labels

        String filePath = FilenameUtils.concat(System.getProperty("user.dir"), "animals");
        File f = new File(filePath);
        List<String> labels = Arrays.asList(f.list());

        // Load in Images to record reader, turn into a single data set to be split
        RecordReader recordReader = new ImageRecordReader(width, height, channels, appendLabels, labels);
        recordReader.initialize(new FileSplit(f));

        // Make a single data set to be normalized and then split
        DataSetIterator allDataIterator = new RecordReaderDataSetIterator(recordReader, numExamples, -1, outputNum);
        DataSet allData = allDataIterator.next();
        allData.normalizeZeroMeanZeroUnitVariance();
        allData.shuffle();

        // Send models and data to cross validation training function
        System.out.println("training...");
        validationTrain(models, allData, nFolds, batchSize, modelParams);

        // Save configuration and parameters to a local file

        //////////  This section would require a dedicated test set to train the best model against using the full dataset

        /*
        String confPath = FilenameUtils.concat(System.getProperty("user.dir"), "AnimalModel-conf.json");
        String paramPath = FilenameUtils.concat(System.getProperty("user.dir"), "AnimalModel-params.bin");

        OutputStream fos = Files.newOutputStream(Paths.get(paramPath));
        DataOutputStream dos = new DataOutputStream(fos);
        Nd4j.write(model.params(), dos);
        dos.flush();
        dos.close();
        FileUtils.writeStringToFile(new File(confPath), model.getLayerWiseConfigurations().toJson());
        */
    }

    ///////////////////////////////////////

    /////////    Helper functions

    ///////////////////////////////////////

    private static int validationTrain(MultiLayerConfiguration[] models, DataSet data, int nFolds, int batchSize,String[] modelParams){
        double maxF1 = 0;
        double maxAvgF1 = 0;
        int maxIndex = 0;
        int maxAvgIndex = 0;
        int[] f1Scores = new int[models.length];
        data.shuffle();
        List<DataSet> folds = data.batchBy(numExamples/nFolds);
        for(int i = 0; i < nFolds; i++){
            DataSet testData = folds.get(i);
            System.out.println(testData.labelCounts());
            List<DataSet> tempData = folds;
            tempData.remove(i);
            DataSet trainData = DataSet.merge(tempData);
            for(int j = 0; j < models.length; j++){
                double currF1 = trainer(models[j],trainData,testData,batchSize);
                f1Scores[j] += currF1;
                if (currF1 > maxF1){
                    maxF1 = currF1;
                    maxIndex = j;
                }
            }
        }
        for (int i = 0; i < models.length; i++){
            f1Scores[i] /= nFolds;
            if (f1Scores[i] > maxAvgF1){
                maxAvgF1 = f1Scores[i];
                maxAvgIndex = i;
            }
        }

        System.out.println("The winner is model"+maxIndex+" with an F1 score of "+maxF1+" and parameters: "+modelParams[maxIndex]);
        System.out.println("The average winner is model"+maxAvgIndex+" with an F1 score of "+maxAvgF1+" and parameters: "+modelParams[maxIndex]);

        return 0;
    }

    private static double trainer(MultiLayerConfiguration conf, DataSet trainData, DataSet testData, int batchSize){
        double f1 = 0;
        int f1Counter = 0;
        List<DataSet> trainIter = trainData.batchBy(batchSize);
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(3));


        while(f1Counter <= 5){
            for(int j = 0; j < trainIter.size(); j++){
                model.fit(trainIter.get(j));
            }
            Evaluation eval = new Evaluation(outputNum);
            eval.eval(testData.getLabels(), model.output(testData.getFeatureMatrix(), Layer.TrainingMode.TEST));
            double currF1 = eval.f1();
            if(currF1 > f1){
                f1 = currF1;
                f1Counter = 0;
            }else f1Counter++;
            System.out.println("Best = "+f1+"\nand current = "+currF1);
        }
        return f1;
    }

    private static MultiLayerConfiguration makeNetwork(double l2Lambda, double dropout) {

        // Configure Convo Neural Net with 3 pairs of convo-pooling layers

        MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .learningRate(0.001)
                .momentum(0.9)
                //.learningRateScoreBasedDecayRate(.01)
                .regularization(true).l2(l2Lambda)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .activation("relu")
                .updater(Updater.NESTEROVS)
                .list(10)
                .layer(0, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(1,1)
                        .nOut(128)
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .padding(1,1)
                        .nOut(256)
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())
                .layer(4, new ConvolutionLayer.Builder(3, 3)
                        .nIn(channels)
                        .stride(1, 1)
                        .padding(1,1)
                        .nOut(512)
                        .build())
                .layer(5, new ConvolutionLayer.Builder(3, 3)
                        .stride(1, 1)
                        .padding(1,1)
                        .biasInit(1)
                        .nOut(512)
                        .build())
                .layer(6, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(3, 3)
                        .stride(2, 2)
                        .build())
                .layer(7, new DenseLayer.Builder()
                        .nOut(2048)
                        .dropOut(dropout)
                        .biasInit(1)
                        .build())
                .layer(8, new DenseLayer.Builder()
                        .nOut(2048)
                        .dropOut(dropout)
                        .biasInit(1)
                        .build())
                .layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false).cnnInputSize(height,width,channels);
        new ConvolutionLayerSetup(builder, width, height, channels);
        MultiLayerConfiguration conf = builder.build();

        return conf;

    }

}