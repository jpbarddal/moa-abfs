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
import moa.classifiers.featureselection.FeatureSelectionClassifier;
import moa.classifiers.featureselection.newfeatureselection.BoostingSelector;
import moa.classifiers.featureselection.newfeatureselection.BoostingSelector2;
import moa.classifiers.trees.HoeffdingTree;
import moa.evaluation.LearningCurve;
import moa.streams.ExampleStream;
import moa.tasks.EvaluateFeatureSelectionStability;

import java.util.ArrayList;
import java.util.HashMap;

public class StabilityExperiments {


    private static final int NUM_INSTANCES_STREAM = 200000;
    private static final int EVALUATION_FREQUENCY =  10000;
    private static final int DRIFT_WINDOW_SIZE = 1000;
    private static final int NUM_CONCEPTS = 3;
    private static final int NUM_IRRELEVANT_FEATURES = 200;

    public static void main(String args[]){
        HashMap<String, ExampleStream> streams = instantiateExperiments();
        HashMap<String, Classifier> selectors = instantiateSelectors();
        String validations[] = new String[]{"Cross-Validation", "Bootstrap-Validation", "Split-Validation"};

        for(String stream : streams.keySet()){
            System.out.println("--> " + stream);
            for(String selector : selectors.keySet()){
                System.out.println("\t\t --> " + selector.toString());
                for(String validation : validations) {
                    EvaluateFeatureSelectionStability task = new EvaluateFeatureSelectionStability();
                    Classifier s = selectors.get(selector);
                    task.selectionOption.setCurrentObject(s.copy());
                    task.instanceLimitOption.setValue(NUM_INSTANCES_STREAM);
                    task.sampleFrequencyOption.setValue(EVALUATION_FREQUENCY);
                    task.memCheckFrequencyOption.setValue(EVALUATION_FREQUENCY);
                    task.validationMethodologyOption.setChosenLabel(validation);
                    task.similarityMetricOption.setChosenLabel("Tanimoto");
                    ExampleStream sCpy = streams.get(stream);
                    sCpy.restart();
                    task.streamOption.setCurrentObject(sCpy);
                    task.dumpFileOption.setValue("./VAL=" + validation + "_" + selector + "-" + stream + ".csv");
                    task.prepareForUse();
                    LearningCurve lc = (LearningCurve) task.doTask();
                }
            }
        }
    }

    private static HashMap<String,Classifier> instantiateSelectors() {
        HashMap<String, Classifier> base = new HashMap<>();
        base.put("NB", new NaiveBayes());
//        base.put("HT", new HoeffdingTree());

        BoostingSelector2 selec = new BoostingSelector2();
        selec.gracePeriodOption.setValue(500);
        selec.selectionStrategyOption.setChosenLabel("MANUAL");
        selec.selectionThresholdOption.setValue(0.01);
        selec.driftDetectionMethodOption.setCurrentObject(new ADWINChangeDetector());
//        BoostingSelector selec = new BoostingSelector();
//        selec.selectionStrategyOption.setChosenLabel("ACC HOEFF");
        selec.prepareForUse();
        selec.resetLearning();

        HashMap<String, Classifier> selectors = new HashMap<>();
        for(String s : base.keySet()){
            FeatureSelectionClassifier fsc = new FeatureSelectionClassifier();
            fsc.selectorOption.setCurrentObject(selec.copy());
            fsc.baseLearnerOption.setCurrentObject(base.get(s));
            fsc.prepareForUse();
            fsc.resetLearning();
            selectors.put("ABFS-" + s, fsc);
        }
        return selectors;
    }

    private static HashMap<String, ExampleStream> instantiateExperiments(){
        FactoryConceptDriftedStreams factory = new FactoryConceptDriftedStreams(NUM_IRRELEVANT_FEATURES,
                NUM_INSTANCES_STREAM, NUM_CONCEPTS, DRIFT_WINDOW_SIZE);
        FactoryConceptDriftedStreams syntheticFactory = new FactoryConceptDriftedStreams(NUM_IRRELEVANT_FEATURES,
                NUM_INSTANCES_STREAM, NUM_CONCEPTS, DRIFT_WINDOW_SIZE);
        FactoryRealWorldStreams factoryReal = new FactoryRealWorldStreams();
        HashMap<String, ExampleStream> all = factory.instantiateAll();
//        HashMap<String, ExampleStream> allReal = factoryReal.instantiateAll();
//        for(String k : allReal.keySet()) all.put(k, allReal.get(k));

        ArrayList<String> toRemove = new ArrayList<String>();
        for(String s : all.keySet()) if (s.contains("(A)")) toRemove.add(s);
        for(String s : toRemove) all.remove(s);

        return all;
    }
}
