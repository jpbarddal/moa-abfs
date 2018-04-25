package moa.classifiers.meta.featureselection;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.SamoaToWekaInstanceConverter;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.Measurement;
import moa.options.ClassOption;
import weka.attributeSelection.InfoGainAttributeEval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.TreeSet;

/**
 *
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @version 1.0
 */
public class OzaBoostingLayer extends AbstractClassifier {

    private DecisionStumpBoosting decisionStump;
    private ChangeDetector detector;
    private boolean changeDetected;
//    private boolean shouldCheckRedundance;
    private int attributeSelected;
    private ArrayList<Integer> redundants;
    private int numMisclassified;
    private int numInstancesSeen;
    private double scms;
    private double swms;

    //buffer used to store instances for redundancy computations
    private Instances buffer;

    // internals for the accuracy decision method    
    private int hitsBeforeSplit;
    private int numInstancesBeforeSplit;
    private int hitsAfterSplit;
    private int numInstancesAfterSplit;
    private boolean testedForAccImprovements;
    private boolean improvedAcc = false;
    private int indexLayer = -1;

    public IntOption gracePeriodOption
            = new IntOption("gracePeriod", 'g', "", 200, 1, 10000);
    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            'c', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");
    public FloatOption selectionThresholdOption
            = new FloatOption("selectionThreshold", 't', "", 0.2, 0.0, 2.0);

    public ClassOption driftDetectionMethodOption
            = new ClassOption("driftDetectionMethod", 'd',
                    "Drift detection method to use.", ChangeDetector.class, "DDM");
    public FlagOption accImprovementsOption
            = new FlagOption("accImprovements", 'a', "");
    public FloatOption minMeritForSplitOption = new FloatOption("minMeritForSplit", 'M',
            "Threshold for minimum merit.", 1e-10, 0.0, 1.0);

    public OzaBoostingLayer() {}

    @Override
    public double[] getVotesForInstance(Instance instnc) {
        try {
            return this.decisionStump.getVotesForInstance(instnc);
        } catch (Exception e) {
            return new double[instnc.numClasses()];
        }
    }

    @Override
    public void resetLearningImpl() {
        this.decisionStump = new DecisionStumpBoosting();
        this.decisionStump.gracePeriodOption.setValue(this.gracePeriodOption.getValue());
        this.decisionStump.splitCriterionOption.setValueViaCLIString(this.splitCriterionOption.getValueAsCLIString());
        this.decisionStump.minMeritForSplitOption.setValue(selectionThresholdOption.getValue());
        this.decisionStump.resetLearning();
        this.decisionStump.prepareForUse();
        this.detector = ((ChangeDetector) getPreparedClassOption(this.driftDetectionMethodOption)).copy();
        this.detector.resetLearning();
        this.detector.prepareForUse();
        this.changeDetected = false;
        this.attributeSelected = -1;
        this.numMisclassified = 0;
        this.numInstancesSeen = 0;
        this.scms = 0.0f;
        this.swms = 0.0f;
        this.testedForAccImprovements = false;
        this.improvedAcc = false;
        this.numInstancesAfterSplit = 0;
        this.numInstancesBeforeSplit = 0;
        this.hitsAfterSplit = 0;
        this.hitsBeforeSplit = 0;
    }

    @Override
    public void trainOnInstanceImpl(Instance instnc) {
        if (this.attributeSelected == -1) {
            //initializes buffer
            if (this.buffer == null) {
                this.buffer = new Instances(instnc.dataset());
            }
            this.buffer.add(instnc);

            this.decisionStump.trainOnInstance(instnc);
            AttributeSplitSuggestion sgt = this.decisionStump.getBestSplit();
            if (sgt != null
                    && sgt.splitTest != null) {
                if (sgt.splitTest.getAttsTestDependsOn() != null) {
                    this.attributeSelected = sgt.splitTest.getAttsTestDependsOn()[0];
//                    String att = instnc.attribute(attributeSelected).name();
//                    double merit = sgt.merit;
//                    double adjustedMerit = sgt.merit * (1.0 / indexLayer);
                    this.clearBuffer();
                }
            }
        } else {
            boolean correctlyClassifies = false;
            /*
            ---------------------------------- DISCLAIMER ----------------------------------
            Depending on how many instances the decision stump has seen during training,
            it is possible that certain values (or ranges or values) haven't been observed,
            and thus, these are not accounted for in the internal counters. In these cases,
            this will raise an Exception, and thus, we will assume that the decision stump
            is unable to correctly predict the class for that instance.
             */
            try {
                correctlyClassifies = this.decisionStump.correctlyClassifies(instnc);
            }catch(Exception e){
                correctlyClassifies = false;
            }
            if (!correctlyClassifies) {
                numMisclassified++;
            }
            numInstancesSeen++;
            this.detector.input(correctlyClassifies ? 0.0 : 1.0);
            if (this.detector.getChange()) {
                this.changeDetected = true;
            }
        }
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return null;
    }

    @Override
    public void getModelDescription(StringBuilder sb, int i) {
    }

    @Override
    public boolean isRandomizable() {
        return false;
    }

    public boolean isChangeDetected() {
        return changeDetected;
    }

    public int getAttributeSelected() {
        return attributeSelected;
    }

    @Override
    public double trainingWeightSeenByModel() {
        return this.decisionStump.trainingWeightSeenByModel();
    }

    public double getEstimation() {
        return detector.getEstimation();
    }

    public double getMerit() {
        return this.decisionStump.getBestSplit().merit;
    }

    public int getNumMisclassified() {
        return numMisclassified;
    }

    public int getNumInstancesSeen() {
        return numInstancesSeen;
    }

    public double getScms() {
        return scms;
    }

    public double getSwms() {
        return swms;
    }

    public void setScms(double scms) {
        this.scms = scms;
    }

    public void setSwms(double swms) {
        this.swms = swms;
    }

//    public void setShouldCheckRedundance(boolean shouldCheckRedundance) {
//        this.shouldCheckRedundance = shouldCheckRedundance;
//    }

    public float getErrorRate() {
        return ((float) numMisclassified) / numInstancesSeen;
    }

    public double getWeight() {
        if(scms + swms == 0) return 0.0;
        double em = this.swms / (this.scms + this.swms);
        if (em == 0.0 || em > 0.5) {
            return 0.0f;
        }
        double bm = em / (1.0 - em);
        return Math.log(1.0 / bm);
    }

    public ArrayList<Integer> getRedundants() {
        return redundants;
    }

    public boolean shouldBeRemoved() {
        /*
         * First condition: we should NOT check for accuracy rates
         * and the weight is below the required threshold
         * */
//        double m = this.getMerit();
//        boolean normal = this.getMerit() < this.selectionThresholdOption.getValue();
//        double adjustedMerit = this.getMerit() * indexLayer;
////        double adjustedMerit = this.getMerit() * Math.pow((double) indexLayer + (indexLayer / (double) this.decisionStump.getAttsScores().size()), 1.0);
//        boolean adjusted = adjustedMerit < this.selectionThresholdOption.getValue();
//        if(adjusted == false){
//            int x = 1;
//        }
//        return adjusted;



//        if (!this.accImprovementsOption.isSet()
//                && this.numInstancesSeen > this.gracePeriodOption.getValue()
//                && this.getWeight() < selectionThresholdOption.getValue()) {
//            return true;
//        /*
//        * Second condition: we have tested for accuracy,
//        * but it has NOT improved
//        * */
//        } else if (this.accImprovementsOption.isSet()
//                && this.isTestedForAccImprovements()
//                && !this.isImprovedAcc()) {
//            return true;
//        }
        return false;
    }

    private void clearBuffer() {
        this.buffer = null;
    }

//    private ArrayList<Integer> findRedundants(TreeSet<DecisionStumpBoosting.FeatureScore> scores) {
//        ArrayList<Integer> redundants = new ArrayList<>();
//
//        SamoaToWekaInstanceConverter cvt = new SamoaToWekaInstanceConverter();
//        weka.core.Instances wekaInstances = cvt.wekaInstances(buffer);
//        InfoGainAttributeEval infogain = new InfoGainAttributeEval();
//        for (DecisionStumpBoosting.FeatureScore s : scores) {
//            // remove all but 's' and attributeSelected
//            weka.core.Instances cpy = new weka.core.Instances(wekaInstances);
//            cpy.setClassIndex(attributeSelected);
//            ArrayList<Integer> toRemove = new ArrayList<>();
//            for (int i = 0; i < cpy.numAttributes(); i++) {
//                if (i != s.getAttIndex() && i != attributeSelected) {
//                    toRemove.add(i);
//                }
//            }
//            Collections.sort(toRemove);
//            Collections.reverse(toRemove);
//            for (Integer i : toRemove) {
//                cpy.deleteAttributeAt(i);
//            }
//            double redun = 0.0;
//            try {
//                cpy.setClassIndex(cpy.numAttributes() - 1);
//                infogain.buildEvaluator(cpy);
//                redun = infogain.evaluateAttribute(0);
//            } catch (Exception e) {
//                System.out.println("ERROR [" + s.getAttIndex() + "] w/ [" + attributeSelected + "]");
//            }
//            if (redun > s.getScore() + 0.1) {
////                System.out.println("RED[" + s.getAttIndex() + "," + attributeSelected + "]");
//                redundants.add(s.getAttIndex());
//            }
//        }
//
//        return redundants;
//    }

    protected void applyDecay(double lambda) {
        this.swms = this.swms * (1.0 - lambda / 2.0);
        this.scms = this.scms * (1.0 - lambda / 2.0);
    }

    public void observeHit(double hit) {
        // if this hasn't been tested already
        if (!isTestedForAccImprovements()) {
            // then we need to check if we are before of after the split
            if (this.attributeSelected == -1) { //before split
                incrementBeforeSplit(hit);
            } else {// after the split
                incrementAfterSplit(hit);
            }
            if (numInstancesAfterSplit >= gracePeriodOption.getValue()) {
                testForAccImprovements();
            }
        }
    }

    private void incrementBeforeSplit(double hit) {
        this.hitsBeforeSplit += hit;
        this.numInstancesBeforeSplit++;
    }

    private void incrementAfterSplit(double hit) {
        this.hitsAfterSplit += hit;
        this.numInstancesAfterSplit++;
    }

    public boolean isTestedForAccImprovements() {
        return testedForAccImprovements;
    }

    private double getHitRateBeforeSplit() {
        if (numInstancesBeforeSplit == 0) {
            return 0.0;
        }
        return ((double) hitsBeforeSplit) / numInstancesBeforeSplit;
    }

    private double getHitRateAfterSplit() {
        if(numInstancesSeen == 0) return Double.NaN;
        return ((double) hitsAfterSplit) / numInstancesAfterSplit;
    }

    public boolean isImprovedAcc() {
        return improvedAcc;
    }

    public double getAccImprovements() {
        double before = getHitRateBeforeSplit();
        double after = getHitRateAfterSplit();
        double diff = after - before;
        return diff;
    }

    public boolean testForAccImprovements() {
        this.testedForAccImprovements = true;
        double diff = getAccImprovements();
        this.improvedAcc = diff >= selectionThresholdOption.getValue();
        if(Double.isNaN(diff)){
            System.err.println("[UNEXPECTED BEHAVIOR] - FOUND A NaN ACCURACY IMPROVEMENT!");
        }
        return this.improvedAcc;
    }

    public void setIndexLayer(int indexLayer) {
        this.indexLayer = indexLayer;
    }
}
