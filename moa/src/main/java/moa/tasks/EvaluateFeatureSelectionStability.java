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

package moa.tasks;

import com.github.javacliparser.FileOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.classifiers.featureselection.AbstractFeatureSelector;
import moa.classifiers.featureselection.FeatureSelectionClassifier;
import moa.core.*;
import moa.evaluation.LearningCurve;
import moa.evaluation.LearningEvaluation;
import moa.options.ClassOption;
import moa.streams.ExampleStream;

import java.io.File;
import java.io.FileOutputStream;
import java.io.PrintStream;
import java.util.Arrays;
import java.util.LinkedList;
import java.util.Random;

// TODO: add a comment to describe the task and also some comments on authors and paper
public class EvaluateFeatureSelectionStability extends ClassificationMainTask {

    public ClassOption selectionOption = new ClassOption("selector", 'S',
            "Feature selection method.", FeatureSelectionClassifier.class,
            "moa.classifiers.featureselection.FeatureSelectionClassifier");

    public ClassOption streamOption = new ClassOption("stream", 's',
            "Stream to learn from.", ExampleStream.class,
            "generators.AssetNegotiationGenerator");

    public IntOption instanceLimitOption = new IntOption("instanceLimit", 'i',
            "Maximum number of instances to test/train on  (-1 = no limit).",
            100000000, -1, Integer.MAX_VALUE);

    public IntOption timeLimitOption = new IntOption("timeLimit", 't',
            "Maximum number of seconds to test/train for (-1 = no limit).", -1,
            -1, Integer.MAX_VALUE);

    public IntOption sampleFrequencyOption = new IntOption("sampleFrequency",
            'f',
            "How many instances between samples of the selection performance.",
            100000, 0, Integer.MAX_VALUE);

    public IntOption memCheckFrequencyOption = new IntOption(
            "memCheckFrequency", 'q',
            "How many instances between memory bound checks.", 100000, 0,
            Integer.MAX_VALUE);

    public FileOption dumpFileOption = new FileOption("dumpFile", 'd',
            "File to append intermediate csv results to.", null, "csv", true);

    public IntOption numFoldsOption = new IntOption("numFolds", 'w',
            "The number of folds (e.g. distributed models) to be used.", 10, 1, Integer.MAX_VALUE);

    public MultiChoiceOption validationMethodologyOption = new MultiChoiceOption(
            "validationMethodology", 'a', "Validation methodology to use.", new String[]{
            "Cross-Validation", "Bootstrap-Validation", "Split-Validation"},
            new String[]{"k-fold distributed Cross Validation",
                    "k-fold distributed Bootstrap Validation",
                    "k-fold distributed Split Validation"
            }, 0);

    public IntOption randomSeedOption = new IntOption("randomSeed", 'r',
            "Seed for random behaviour of the task.", 1);

    public MultiChoiceOption similarityMetricOption = new MultiChoiceOption(
            "similarityMetric", 'm', "Similarity metric to use.", new String[]{
            "Tanimoto", "Pearson"},
            new String[]{"Tanimoto coefficient.",
                    "Pearson coefficient."
            }, 1);


    // TODO: add the citation for our cool paper below
    /**
     * This task performs a Prequential Cross-validation-like procedure
     * to compute the stability of a feature selection method.
     * Details about the proposed method can be found here in the paper below.
     * <p></p>
     *
     * @param monitor    the TaskMonitor to use
     * @param repository the ObjectRepository to use
     * @return the learning curve
     */
    @Override
    protected Object doMainTask(TaskMonitor monitor, ObjectRepository repository) {

        Random random = new Random(this.randomSeedOption.getValue());
        ExampleStream stream = (ExampleStream) getPreparedClassOption(this.streamOption);

        FeatureSelectionClassifier[] selectors = new FeatureSelectionClassifier[this.numFoldsOption.getValue()];
        FeatureSelectionClassifier baseSelector = (FeatureSelectionClassifier) getPreparedClassOption(this.selectionOption);
        baseSelector.setComputingStability();
        baseSelector.resetLearning();

        for (int i = 0; i < selectors.length; i++) {
            selectors[i] = (FeatureSelectionClassifier) baseSelector.copy();
            selectors[i].setModelContext(stream.getHeader());
        }

        LearningCurve learningCurve = new LearningCurve(
                "learning evaluation instances");
        int maxInstances = this.instanceLimitOption.getValue();
        long instancesProcessed = 0;
        int maxSeconds = this.timeLimitOption.getValue();
        int secondsElapsed = 0;
        monitor.setCurrentActivity("Evaluating learner...", -1.0);

        File dumpFile = this.dumpFileOption.getFile();
        PrintStream immediateResultStream = null;
        if (dumpFile != null) {
            try {
                if (dumpFile.exists()) {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile, true), true);
                } else {
                    immediateResultStream = new PrintStream(
                            new FileOutputStream(dumpFile), true);
                }
            } catch (Exception ex) {
                throw new RuntimeException(
                        "Unable to open immediate result file: " + dumpFile, ex);
            }
        }

        boolean firstDump = true;
        boolean preciseCPUTiming = TimingUtils.enablePreciseTiming();
        long evaluateStartTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
        long lastEvaluateStartTime = evaluateStartTime;
        double RAMHours = 0.0;
        while (stream.hasMoreInstances()
                && ((maxInstances < 0) || (instancesProcessed < maxInstances))
                && ((maxSeconds < 0) || (secondsElapsed < maxSeconds))) {
            Example trainInst = stream.nextInstance();

            for (int i = 0; i < selectors.length; i++) {
                int k = 1;
                switch (this.validationMethodologyOption.getChosenIndex()) {
                    case 0: //Cross-Validation;
                        k = instancesProcessed % selectors.length == i ? 0: 1; //Test all except one
                        break;
                    case 1: //Bootstrap;
                        k = MiscUtils.poisson(1, random);
                        break;
                    case 2: //Split-Validation;
                        k = instancesProcessed % selectors.length == i ? 1: 0; //Test only one
                        break;
                }
                if (k > 0) {
                    Example weightedInst = trainInst.copy();
                    weightedInst.setWeight(trainInst.weight() * k);
                    selectors[i].trainOnInstance((Instance) weightedInst.getData());
                }

            }

            instancesProcessed++;
            if (instancesProcessed % this.sampleFrequencyOption.getValue() == 0
                    || !stream.hasMoreInstances()) {
                long evaluateTime = TimingUtils.getNanoCPUTimeOfCurrentThread();
                double time = TimingUtils.nanoTimeToSeconds(evaluateTime - evaluateStartTime);
                double timeIncrement = TimingUtils.nanoTimeToSeconds(evaluateTime - lastEvaluateStartTime);

                for (int i = 0; i < selectors.length; i++) {
                    double RAMHoursIncrement = selectors[i].measureByteSize() / (1024.0 * 1024.0 * 1024.0); //GBs
                    RAMHoursIncrement *= (timeIncrement / 3600.0); //Hours
                    RAMHours += RAMHoursIncrement;
                }

                // obtains the number of features excluding the class attribute
                long fullSetSize = ((Instance) trainInst.getData()).numAttributes() - 1;
                lastEvaluateStartTime = evaluateTime;
                double stabilityResults[] = stability(selectors, fullSetSize);
                Measurement m[] = new Measurement[]{
                        new Measurement(
                                "learning evaluation instances",
                                instancesProcessed),
                        new Measurement("stability", stabilityResults[0]),
                                new Measurement("stddev. stability", stabilityResults[1]),
                        new Measurement(
                                "evaluation time ("
                                        + (preciseCPUTiming ? "cpu "
                                        : "") + "seconds)",
                                time),
                        new Measurement(
                                "model cost (RAM-Hours)",
                                RAMHours)
                };

                Measurement mSelectors[] = obtainAverageMeasurements(selectors);

                Measurement[] result = Arrays.copyOf(m, m.length + mSelectors.length);
                System.arraycopy(mSelectors, 0, result, m.length, mSelectors.length);

                learningCurve.insertEntry(new LearningEvaluation(result));

                if (immediateResultStream != null) {
                    if (firstDump) {
                        immediateResultStream.println(learningCurve.headerToString());
                        firstDump = false;
                    }
                    immediateResultStream.println(learningCurve.entryToString(learningCurve.numEntries() - 1));
                    immediateResultStream.flush();
                }
            }
            if (instancesProcessed % INSTANCES_BETWEEN_MONITOR_UPDATES == 0) {
                if (monitor.taskShouldAbort()) {
                    return null;
                }
                long estimatedRemainingInstances = stream.estimatedRemainingInstances();
                if (maxInstances > 0) {
                    long maxRemaining = maxInstances - instancesProcessed;
                    if ((estimatedRemainingInstances < 0)
                            || (maxRemaining < estimatedRemainingInstances)) {
                        estimatedRemainingInstances = maxRemaining;
                    }
                }
                monitor.setCurrentActivityFractionComplete(estimatedRemainingInstances < 0 ? -1.0
                        : (double) instancesProcessed / (double) (instancesProcessed + estimatedRemainingInstances));
                if (monitor.resultPreviewRequested()) {
                    monitor.setLatestResultPreview(learningCurve.copy());
                }
                secondsElapsed = (int) TimingUtils.nanoTimeToSeconds(TimingUtils.getNanoCPUTimeOfCurrentThread()
                        - evaluateStartTime);
            }
        }
        if (immediateResultStream != null) {
            immediateResultStream.close();
        }
        return learningCurve;

    }

    /**
     * Computes the average of the measurements obtained by each of the selectors.
     *
     * @param selectors
     * @return an array with the avg measurements.
     */
    private Measurement[] obtainAverageMeasurements(FeatureSelectionClassifier[] selectors) {
        double v[] = null;
        String h[] = null;
        for(FeatureSelectionClassifier s : selectors){
            Measurement ms[] = s.getModelMeasurements();
            if(v == null){
                v = new double[ms.length];
                h = new String[ms.length];
            }

            for(int i = 0; i < ms.length; i++){
                v[i] += ms[i].getValue();
                if(h[i] == null){
                    h[i] = "[selector] avg. " + ms[i].getName();
                }
            }
        }

        //computes the avg creates the final vector
        Measurement finalMeasurements[] = new Measurement[v.length];
        for(int i = 0; i < v.length; i++){
            finalMeasurements[i] = new Measurement(h[i], v[i]/selectors.length);
        }

        return finalMeasurements;
    }

    /**
     * This task results in a learning curve.
     * It will present the stability obtained along the stream with the following format:
     * Instances seen, stability
     *
     * @return a learning curve with the stability results
     */
    @Override
    public Class<?> getTaskResultType() {
        return LearningCurve.class;
    }


    /**
     * Computes the stability across all folds.
     * The stability depends on the similarity metric adopted.
     *
     * @return the stability
     */
    private double[] stability(FeatureSelectionClassifier selectors[], long fullSetSize){
        LinkedList<int[]> selections = new LinkedList<>();

        // retrieves the subsets of selected features
        for(FeatureSelectionClassifier selector : selectors){
            int s[] = selector.getSelectedFeatures();
            if(s != null && s.length > 0) selections.add(s);
        }

        // In total, we have n * (n - 1)/2 pairs, where n is the number of folds (selectors)
        double similarities[] = new double[selections.size() * (selections.size() - 1) / 2];
        int is = 0;
        for(int i = 0; i < selections.size(); i++){
            for(int j= i + 1; j < selections.size(); j++){
                // for each pair of selectors, calculate a similarity index
                similarities[is] = similarity(selections.get(i), selections.get(j), fullSetSize);
                is++;
            }
        }

        if (similarities.length > 0){
            double mean = FeatureSelectionUtils.mean(similarities);
            return new double[]{mean, FeatureSelectionUtils.stddev(similarities, mean)};
        } else {
            return new double[]{Double.NaN, Double.NaN};
        }
    }

    /**
     * Calculates the similarity between two subsets of features.
     * The similarity is given by either Tanimoto or Pearson coefficients.
     *
     * @param sA the first subset of features
     * @param sB the second subset of features
     * @param sizeFullSet the size of the original set of features
     * @return the similarity between two subsets of features
     */
    private double similarity(int sA[], int sB[], long sizeFullSet){
        double similarity;
        if (similarityMetricOption.getChosenLabel().equals("Tanimoto")){
            similarity = tanimoto(sA, sB);
        }else if (similarityMetricOption.getChosenLabel().equals("Pearson")){
            similarity = pearson(sA, sB, sizeFullSet);
        }else {
            throw new IllegalArgumentException("The similarity metric should be either 'Tanimoto' or 'Pearson'.");
        }
        return similarity;
    }

    /**
     * Tanimoto coefficient.
     *
     * @param sA the first set
     * @param sB the second set
     * @return the tanimoto coefficient between the arrays
     */
    private double tanimoto(int sA[], int sB[]){
        // aka Jaccard index
        long sizeA = sA.length;
        long sizeB = sB.length;
        long intersection = FeatureSelectionUtils.intersection(sA, sB);
        double tanimoto = intersection / ((double) sizeA + sizeB - intersection);
        return tanimoto;
    }

    /**
     * Pearson coefficient for similarity between arrays
     *
     * @param sA the first set
     * @param sB the second set
     * @param sizeFullSet the number of features in the entire set of features
     * @return the pearson coefficient between the arrays
     */
    private double pearson(int sA[], int sB[], long sizeFullSet){
        double meanA = sA.length / (double) sizeFullSet;
        double meanB = sB.length / (double) sizeFullSet;

        double pt1 = 0.0;
        double pt2 = 0.0;
        double pt3 = 0.0;

        for (int i = 0; i < sizeFullSet; i++){
            double difA = FeatureSelectionUtils.contains(i, sA) ? 1 - meanA : 0 - meanA;
            double difB = FeatureSelectionUtils.contains(i, sB) ? 1 - meanB : 0 - meanB;

            pt1 += Math.abs(difA * difB);
            pt2 += difA * difA;
            pt3 += difB * difB;
        }
        pt2 = Math.sqrt(pt2);
        pt3 = Math.sqrt(pt3);
        double pearson = pt1 / (pt2 * pt3);
        return pearson;
    }

}
