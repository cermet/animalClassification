import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.image.recordreader.ImageRecordReader;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.Layer;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.weights.HistogramIterationListener;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.DataOutputStream;
import java.io.File;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Random;


public class Classifier {

        public static void main (String[] args) throws IOException, InterruptedException {

            // Initialized parameters

            int seed = 123;
            int height = 50;
            int width = 50;
            int wByH = width * height;
            int numExamples = 403;
            int outputNum = 4;
            int batchSize = 20;
            int listenerFreq = 1;
            boolean appendLabels = true;
            int iterations = 2;
            int epochs = 30;
            double splitTrainPerc = .8;

            // File and labels

            String filePath = FilenameUtils.concat(System.getProperty("user.dir"), "animals");
            File f = new File(filePath);
            List<String> labels = Arrays.asList(f.list());

            // Load in Images to record reader, turn into a single data set to be split

            RecordReader recordReader = new ImageRecordReader(width, height, appendLabels, labels);
            recordReader.initialize(new FileSplit(f));

            // Make a single data set to be normalized and then split

            DataSetIterator allDataIterator = new RecordReaderDataSetIterator(recordReader, numExamples, wByH, outputNum);
            DataSet allData = allDataIterator.next();
            allData.normalizeZeroMeanZeroUnitVariance();
            allData.shuffle();
            SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(splitTrainPerc);
            DataSet train = testAndTrain.getTrain();
            List<DataSet> trainIter = train.batchBy(batchSize);
            DataSet test = testAndTrain.getTest();

            System.out.println(train.labelCounts());
            System.out.println(test.labelCounts());

            // Configure Convo Neural Net with 3 pairs of convo-pooling layers

            MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .iterations(iterations)
                    .learningRate(0.0001)
                    .regularization(true).dropOut(.9).l2(.0001)
                    .weightInit(WeightInit.XAVIER)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .updater(Updater.NESTEROVS).momentum(0.9)
                    .list(6)
                    .layer(0, new ConvolutionLayer.Builder(5, 5)
                            .nIn(wByH)
                            .stride(1, 1)
                            .nOut(50)
                            .activation("identity")
                            .build())
                    .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
                    .layer(2, new ConvolutionLayer.Builder(5, 5)
                            .nIn(wByH)
                            .stride(1, 1)
                            .nOut(100)
                            .activation("identity")
                            .build())
                    .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
                            .kernelSize(2, 2)
                            .stride(2, 2)
                            .build())
//                    .layer(4, new ConvolutionLayer.Builder(5, 5)
//                            .nIn(wByH)
//                            .stride(1, 1)
//                            .nOut(150)
//                            .activation("identity")
//                            .build())
//                    .layer(5, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG)
//                            .kernelSize(2, 2)
//                            .stride(2, 2)
//                            .build())
                    .layer(4, new DenseLayer.Builder().activation("relu")
                            .nOut(1000).build())
                    .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                            .nOut(outputNum)
                            .activation("softmax")
                            .build())
                    .backprop(true).pretrain(false);
            new ConvolutionLayerSetup(builder, width, height, 1);

            // Build configuration into a model for training

            MultiLayerConfiguration conf = builder.build();
            MultiLayerNetwork model = new MultiLayerNetwork(conf);
            model.init();

            // Create a histogram listener to visualize learning

           model.setListeners(new ScoreIterationListener(listenerFreq));

            // Train model on defined training data for the defined number of epochs
            for (int i = 0; i < epochs; i++) {
                for(int j = 0; j < trainIter.size(); j++){
                    model.fit(trainIter.get(j));
                }
                Evaluation eval = new Evaluation(outputNum);
                eval.eval(test.getLabels(), model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));
                System.out.println(eval.stats());
            }

            // Evaluate model based on defined testing set and print results

            Evaluation eval = new Evaluation(outputNum);
            eval.eval(test.getLabels(), model.output(test.getFeatureMatrix(), Layer.TrainingMode.TEST));
            System.out.println(eval.stats());

            // Save configuration and parameters to a local file

            String confPath = FilenameUtils.concat(System.getProperty("user.dir"), "AnimalModel-conf.json");
            String paramPath = FilenameUtils.concat(System.getProperty("user.dir"), "AnimalModel-params.bin");

            OutputStream fos = Files.newOutputStream(Paths.get(paramPath));
            DataOutputStream dos = new DataOutputStream(fos);
            Nd4j.write(model.params(), dos);
            dos.flush();
            dos.close();
            FileUtils.writeStringToFile(new File(confPath), model.getLayerWiseConfigurations().toJson());

        }

}