/*
 * Copyright (c) 2017.
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Heitor Murilo Gomes (heitor.gomes@telecom-paristech.fr)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package moa.experiments;

import moa.classifiers.Classifier;
import moa.classifiers.bayes.NaiveBayes;
import moa.classifiers.core.driftdetection.ADWINChangeDetector;
import moa.classifiers.core.driftdetection.AbstractChangeDetector;
import moa.classifiers.core.driftdetection.HDDM_A_Test;
import moa.classifiers.core.driftdetection.HDDM_W_Test;
import moa.classifiers.featureselection.FeatureSelectionClassifier;
import moa.classifiers.lazy.kNN;
import moa.classifiers.meta.featureselection.BoostingSelector;
import moa.classifiers.meta.featureselection.GoldStandardSelector;
import moa.classifiers.trees.HoeffdingAdaptiveTree;
import moa.classifiers.trees.HoeffdingTree;
import moa.evaluation.BasicClassificationPerformanceEvaluator;
import moa.evaluation.LearningPerformanceEvaluator;
import moa.evaluation.WindowClassificationPerformanceEvaluator;
import moa.evaluation.preview.LearningCurve;
import moa.streams.ArffFileStream;
import moa.streams.ExampleStream;
import moa.tasks.EvaluatePrequential;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.HashMap;
import java.util.Map;

/**
 *
 * To run:
 * nohup java -Xms20g -Xmx20g -XX:-UseGCOverheadLimit -cp "moa-pom.jar:./lib/*" -javaagent:./sizeofag.jar moa.experiments.NewFeatureSelectionExperiments > log.log 2> error.log &
 */
public class NewFeatureSelectionExperiments {

    private static final int NUM_INSTANCES_STREAM = 200000;
    private static final int EVALUATION_FREQUENCY =   2000;
    private static final int DRIFT_WINDOW_SIZE = 5000;
    private static final int NUM_CONCEPTS = 3;
    private static final int NUM_IRRELEVANT_FEATURES = 500;
    private static final String EVALUATOR_TYPE = "WINDOWAUC";
    private static final String EXPERIMENT_TYPE = "BOOSTING";

    public static void main(String args[]) throws IOException {
        System.out.println("Loading checkpoint...");
        String checkpoint[] = null;

        System.out.print("Preparing folders for outputting results...");
        prepareFolder(checkpoint == null);

        // Prepares a buffer to output the summary file
        PrintWriter writer = new PrintWriter(new FileOutputStream("./summary.csv", true));
        if (checkpoint == null) {
            writer.println("Stream,Classifier,"
                    + (EXPERIMENT_TYPE.equals("BOOSTING") ? ",,,,,," : "")
                    + "Avg Accuracy, "
                    + "Avg AUC, CPU Time, "
                    + "RAM-Hours, "
                    + "AVG Recall Relevant, AVG Complement of Complexity Penalty, "
                    + "AVG SA, AVG # selected, AVG pct selected");
        }
        System.out.println("OK!");

        System.out.print("Instantiating all classifiers...");
        HashMap<String, Classifier> classifiers = instantiateClassifiers();
        System.out.println("OK!");
        System.out.print("Instantiating all experiments...");
        HashMap<String, ExampleStream> streams = instantiateStreams();
        System.out.println("OK!");

        System.out.println("\n\n\n");
        System.out.println("========= RUNNING  EXPERIMENTS =========\n");
        int indexExperiment = 1;
        int numExperiments = streams.size() * classifiers.size();
        System.out.println(classifiers.size() + " classifiers are instantiated.");
        System.out.println(streams.size() + " streams are instantiated.");
        System.out.println(numExperiments + " experiments are starting... NOW!");
        for (String strStream : streams.keySet()) {
            ExampleStream s;
            if(streams.get(strStream) instanceof ArffFileStream){
                s = (ExampleStream) streams.get(strStream);
            }else{
                s = (ExampleStream) streams.get(strStream).copy();
            }
            System.out.println("\n\n\n\n\n\n- Running: " + strStream + "\t(" + indexExperiment + "/" + (streams.size() * classifiers.size()) + ")");
            for (String strClassifier : classifiers.keySet()) {
                Classifier c = classifiers.get(strClassifier).copy();
                System.out.print("--" + strClassifier + "\t");
                appendResults(writer, runExperiment(strClassifier, c, strStream, s));
                System.out.println(((float) 100 * indexExperiment / (streams.size() * classifiers.size())) + "% overall complete...\n");
                indexExperiment++;
            }
        }

        System.out.println("\n========= EXPERIMENTS COMPLETE =========");
    }

    private static String runExperiment(String strClassifier, Classifier c,
                                        String strStream, ExampleStream s) {
        // prepares the stream and the classifier for execution        
        c.resetLearning();
        c.prepareForUse();
        s.restart();

        // prepares the filename for outputting raw results
        String filename = prepareFileName(strClassifier, strStream);
        eraseIfExists("./results/" + filename);
        eraseIfExists("./debug/" + filename);

        //prepares the evaluator
        LearningPerformanceEvaluator evaluator = null;
        if (EVALUATOR_TYPE.equals("BASIC")) {
            evaluator = new BasicClassificationPerformanceEvaluator();
        } else {
            evaluator
                    = new WindowClassificationPerformanceEvaluator();
        }
        evaluator.reset();

        // runs the experiment  
        EvaluatePrequential prequential = new EvaluatePrequential();
        prequential.evaluatorOption.setCurrentObject(evaluator.copy());
        prequential.instanceLimitOption.setValue(NUM_INSTANCES_STREAM);
        int sampleFreq = EVALUATOR_TYPE.equals("BASIC")
                ? NUM_INSTANCES_STREAM : EVALUATION_FREQUENCY;
        prequential.sampleFrequencyOption.setValue(sampleFreq);
        prequential.dumpFileOption.setValue("./results/" + filename);
        if(s instanceof ArffFileStream){
            s.restart();
            prequential.streamOption.setCurrentObject(s);
        }else{
            prequential.streamOption.setCurrentObject(s.copy());
        }
        prequential.learnerOption.setCurrentObject(c.copy());
        prequential.prepareForUse();
        LearningCurve lc = (LearningCurve) prequential.doTask();

        // extracts the final results for summary
        // {avgAccuracy, avgAUC, cpuTime, ramHours, avgRecallRelevant, 
        //  avgPenaltyIrrelevant, avgSA, avgNumSelected, avgPctSelected};
        double rs[] = getFinalValuesForExperiment(lc);
        double avgAccuracy = rs[0];
        double avgAUC = rs[1];
        double cpuTime = rs[2];
        double ramHours = rs[3];
        double avgRecallRelevant = rs[4];
        double avgPenaltyIrrelevant = rs[5];
        double avgSA = rs[6];
        double avgNumSelected = rs[7];
        double avgPctSelected = rs[8];

        // RETURNS 
        return strStream + "," + strClassifier + ","
                + (!strClassifier.contains("ABFS") && EXPERIMENT_TYPE.equals("BOOSTING") ? ",,,,,," : "")
                + avgAccuracy + ","
                + avgAUC + ","
                + cpuTime + "," + ramHours + ","
                + avgRecallRelevant + ","
                + avgPenaltyIrrelevant + ","
                + avgSA + ","
                + avgNumSelected + ","
                + avgPctSelected;
    }

    ///////////////////////////////////////
    // METHODS TO PREPARE THE EXPERIMENT // 
    ///////////////////////////////////////
    public static HashMap<String, Classifier> instantiateClassifiers() {
        HashMap<String, Classifier> classifiers = new HashMap<>();

        // Base learners
        classifiers.putAll(instantiateBaseLearners());

        // Base Boosting-selected learners
        classifiers.putAll(instantiateBoostingSelectedLearners());

        // Oracle selector
        classifiers.putAll(instantiateOracleSelectors());

        return classifiers;
    }

    private static Map<String, Classifier> instantiateOracleSelectors() {
        // the base learners
        HashMap<String, Classifier> baseLearners = instantiateBaseLearners();
        // the return hashmap
        HashMap<String, Classifier> ret = new HashMap<>();

        for(String base : baseLearners.keySet()) {
            FeatureSelectionClassifier selector = new FeatureSelectionClassifier();
            GoldStandardSelector oracle = new GoldStandardSelector();
            oracle.resetOnFeatureChangeOption.set();
            oracle.prepareForUse();
            oracle.resetLearning();
            selector.baseLearnerOption.setCurrentObject(baseLearners.get(base));
            selector.selectorOption.setCurrentObject(oracle);
            selector.prepareForUse();
            ret.put("ORACLE-" + base, selector);
        }

        return ret;
    }

    public static HashMap<String, Classifier> instantiateBaseLearners() {
        HashMap<String, Classifier> classifiers = new HashMap<>();

        // Hoeffding Tree
        HoeffdingTree tree = new HoeffdingTree();
        tree.prepareForUse();
        classifiers.put("HT", tree);

        // Naive Bayes
        NaiveBayes bayes = new NaiveBayes();
        bayes.prepareForUse();
        classifiers.put("NB", bayes);

        // kNN
        kNN knn = new kNN();
        knn.limitOption.setValue(500);
        knn.prepareForUse();
        classifiers.put("KNN", knn);

        // Hoeffding Adaptive Tree (HAT)
        HoeffdingAdaptiveTree adapTree = new HoeffdingAdaptiveTree();
        adapTree.prepareForUse();
        classifiers.put("HAT", adapTree);

        return classifiers;
    }

    public static HashMap<String, Classifier> instantiateBoostingSelectedLearners() {
        // base learners
        HashMap<String, Classifier> baseLearners = instantiateBaseLearners();

        // boosted-selected classifiers
        HashMap<String, Classifier> classifiers = new HashMap<>();

        // parameters
        int gps[] = new int[]{100, 200, 500, 1000};
        String selectionStrategies[] = new String[]{"MANUAL"};
        double thresholds[] = new double[]{0.01, 0.05, 0.1};
        AbstractChangeDetector detectors[] = new AbstractChangeDetector[]{new ADWINChangeDetector(),
                new HDDM_A_Test(), new HDDM_W_Test()};
        for(String classif : baseLearners.keySet()){
            for(String selectionStrat : selectionStrategies){
                for(AbstractChangeDetector cd : detectors) {
                    for (int gp : gps) {
                        for (double threshold : thresholds) {
                            FeatureSelectionClassifier abstractSelector = new FeatureSelectionClassifier();
                            BoostingSelector selector = new BoostingSelector();
                            selector.verboseOption.set();
                            selector.gracePeriodOption.setValue(gp);
                            selector.driftDetectionMethodOption.setCurrentObject(cd);
                            selector.selectionStrategyOption.setChosenLabel(selectionStrat);
                            selector.selectionThresholdOption.setValue(threshold);
                            abstractSelector.selectorOption.setCurrentObject(selector);
                            abstractSelector.baseLearnerOption.setCurrentObject(baseLearners.get(classif).copy());
                            abstractSelector.prepareForUse();
                            String config = prepareBoostingConfig(selector);
                            config = config.replace("?", classif);
                            classifiers.put(config, abstractSelector);
                        }
                    }
                }
            }
        }
        return classifiers;
    }

    private static String prepareBoostingConfig(BoostingSelector selector) {
        String str = "";
        str += "ABFS,";
        str += "gp=" + selector.gracePeriodOption.getValue() + ",";
        str += "learner=?,";
        str += "split=" + selector.splitCriterionOption.getValueAsCLIString() + ",";
        str += "selection=" + selector.selectionStrategyOption.getChosenLabel() + ",";
        if(selector.selectionStrategyOption.getChosenLabel().equals("HOEFF")){
            str += "threshold=HOEFF,";
        }else if(selector.selectionStrategyOption.getChosenLabel().equals("CHEBYSHEV")){
            str += "threshold=CHEB,";
        }else{
            str += "threshold=" + selector.selectionThresholdOption.getValue() + ",";
        }
        str += "dd=" + selector.driftDetectionMethodOption.getValueAsCLIString();
        return str;
    }

    public static HashMap<String, ExampleStream> instantiateStreams() {
        HashMap<String, ExampleStream> streams = new HashMap<>();
        FactoryConceptDriftedStreams factory
                = new FactoryConceptDriftedStreams(NUM_IRRELEVANT_FEATURES,
                NUM_INSTANCES_STREAM,
                NUM_CONCEPTS,
                DRIFT_WINDOW_SIZE);
        streams.putAll(factory.instantiateAGRAWAL());
        streams.putAll(factory.instantiateASSETS());
        streams.putAll(factory.instantiateBG1());
        streams.putAll(factory.instantiateBG2());
        streams.putAll(factory.instantiateBG3());
        streams.putAll(factory.instantiateRTG());
        streams.putAll(factory.instantiateSEA());

        FactoryRealWorldStreams realFactory = new FactoryRealWorldStreams();
        streams.putAll(realFactory.instantiateAll());

        return streams;
    }

    ////////////////////////////////////////////////////////////////
    // AUXILIAR METHOD TO EXTRACT RESULTS FROM THE LEARNING CURVE //
    ////////////////////////////////////////////////////////////////
    private static double[] getFinalValuesForExperiment(LearningCurve lc) {
        int indexAcc = -1;
        int indexAUC = -1;
        int indexCpuTime = -1;
        int indexRamHours = -1;
        int indexRecallRelevant = -1;
        int indexPenaltyIrrelevant = -1;
        int indexSA = -1;
        int indexNumSelected = -1;
        int indexPctSelected = -1;
        int index = 0;

        for (String s : lc.headerToString().split(",")) {
            if (s.contains("classifications correct")) {
                indexAcc = index;
            } else if (s.contains("AUC")) {
                indexAUC = index;
            } else if (s.contains("time")) {
                indexCpuTime = index;
            } else if (s.contains("RAM-Hours")) {
                indexRamHours = index;
            } else if (s.contains("RRF")) {
                indexRecallRelevant = index;
            } else if (s.contains("CCP")) {
                indexPenaltyIrrelevant = index;
            } else if (s.contains("Selection Accuracy")) {
                indexSA = index;
            } else if (s.contains("# of features selected")) {
                indexNumSelected = index;
            } else if (s.contains("pct of features selected")) {
                indexPctSelected = index;
            }
            index++;
        }

        // Accuracy, AUC and feature selection metrics must be averaged
        double avgAccuracy = 0.0;
        double avgAUC = 0.0;
        double avgRecallRelevant = 0.0;
        double avgPenaltyIrrelevant = 0.0;
        double avgSA = 0.0;
        double avgNumSelected = 0.0;
        double avgPctSelected = 0.0;
        for (int entry = 0; entry < lc.numEntries(); entry++) {
            avgAccuracy += lc.getMeasurement(entry, indexAcc);
            avgAUC += (indexAUC != -1 ? lc.getMeasurement(entry, indexAUC) : 0.0);
            avgRecallRelevant += (indexRecallRelevant != -1 ? lc.getMeasurement(entry, indexRecallRelevant) : 0.0);
            avgPenaltyIrrelevant += (indexPenaltyIrrelevant != -1 ? lc.getMeasurement(entry, indexPenaltyIrrelevant) : 0.0);
            avgSA += (indexSA != -1 ? lc.getMeasurement(entry, indexSA) : 0.0);
            avgNumSelected += (indexNumSelected != -1 ? lc.getMeasurement(entry, indexNumSelected) : 0.0);
            avgPctSelected += (indexPctSelected != -1 ? lc.getMeasurement(entry, indexPctSelected) : 0.0);
        }
        avgAccuracy /= lc.numEntries();
        avgAUC /= lc.numEntries();
        avgRecallRelevant /= lc.numEntries();
        avgPenaltyIrrelevant /= lc.numEntries();
        avgSA /= lc.numEntries();
        avgNumSelected /= lc.numEntries();
        avgPctSelected /= lc.numEntries();

        // but both cpu time and ram hours are only the final values obtained
        // since they represent the processing of the entire stream
        double cpuTime = lc.getMeasurement(lc.numEntries() - 1, indexCpuTime);
        double ramHours = lc.getMeasurement(lc.numEntries() - 1, indexRamHours);

        return new double[]{avgAccuracy, avgAUC, cpuTime, ramHours,
                avgRecallRelevant, avgPenaltyIrrelevant, avgSA,
                avgNumSelected, avgPctSelected};
    }

    /////////////////////////////////////////////
    // AUXILIAR METHODS TO SET UP OUTPUT FILES //
    /////////////////////////////////////////////
    private static void prepareFolder(boolean reset) {
        String paths[] = new String[]{"./results/", "./debug/"};
        File summary = new File("./summary.csv");
        if (summary.exists()) {
            summary.delete();
        }
        for (String path : paths) {
            File folder = new File(path);
            File listOfFiles[];
            if (folder.exists()) {
                listOfFiles = folder.listFiles();
                if (reset) {
                    for (int i = 0; i < listOfFiles.length; i++) {
                        if (listOfFiles[i].isFile()) {
                            if (listOfFiles[i].getName().endsWith(".csv")) {
                                listOfFiles[i].delete();
                            }
                        }
                    }
                }
            } else {
                folder.mkdir();
            }
        }
    }

    private static String prepareFileName(String strClassifier, String strStream) {
        String filename = strStream + "_" + strClassifier + ".csv";
        filename = filename.trim();
        filename = filename.replace("-", "_").replace(" ", "_");
        return filename;
    }

    private static void appendResults(PrintWriter writer, String line) {
        writer.println(line);
        writer.flush();
    }

    private static void eraseIfExists(String string) {
        File file = new File(string);
        if (file.exists()) {
            file.delete();
        }
    }

}
