/*
 * Copyright (c) 2017.
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
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

package moa.classifiers.meta.featureselection;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.MultiChoiceOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.Instances;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.core.driftdetection.ChangeDetector;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.core.ObjectRepository;
import moa.options.ClassOption;
import moa.tasks.TaskMonitor;

import java.util.ArrayList;
import java.util.Collections;

public class BoostingSelector extends AbstractFeatureSelector {

    /**
     * The grace period for selecting a feature.
     */
    public IntOption gracePeriodOption
            = new IntOption("gracePeriod", 'g',
            "The grace period for selecting a feature.", 200, 1, 1000);

    /**
     * Split criterion to use.
     */
    public ClassOption splitCriterionOption = new ClassOption("splitCriterion",
            'c', "Split criterion to use.", SplitCriterion.class,
            "InfoGainSplitCriterion");
    /**
     * Selection strategy.
     */
    public MultiChoiceOption selectionStrategyOption
            = new MultiChoiceOption("selectionStrategy", 's', "",
            new String[]{"MANUAL", "HOEFF", "CHEBYSHEV"},
            new String[]{"MANUAL", "HOEFF", "CHEBYSHEV"}, 0);

    /**
     * Selection threshold.
     */
    public FloatOption selectionThresholdOption
            = new FloatOption("selectionThreshold", 't', "", 0.2, 0.0, 2.0);

    /**
     * Drift detection method.
     */
    public ClassOption driftDetectionMethodOption
            = new ClassOption("driftDetectionMethod", 'd',
            "Drift detection method to use.",
            ChangeDetector.class, "ADWINChangeDetector");

    /**
     * Verbose.
     */
    public FlagOption verboseOption = new FlagOption("verbose", 'v', "");

    public FlagOption deleteOption = new FlagOption("delete", 'D', "");

    /**
     * Boosting layers
     */
    ArrayList<OzaBoostingLayer> layers;

    /**
     * Selected features
     */
    ArrayList<Integer> selectedFeatures;

    /**
     * Unselected features
     */
    ArrayList<Integer> unselectedFeatures;

    /**
     * Window of instances
     */
    private Instances window;

    /**
     * A counter for instances observed
     */
    private int instancesSeen;

    /**
     * A flag for changes in the set of selected features.
     */
    private boolean selectedChanged;

    /**
     * A cache for the selected subset of features.
     */
    private int currentlySelected[];

    /**
     * Method responsible for returning the selected subset of features.
     *
     * @return the indices of the selected features
     */
    @Override
    public int[] getSelectedFeatures() {
        if(selectedChanged) {
            if (selectedFeatures != null && selectedFeatures.size() > 0) {
                currentlySelected = new int[selectedFeatures.size()];
                Collections.sort(selectedFeatures);
                int index = 0;
                for (int v : selectedFeatures) {
                    currentlySelected[index] = v;
                    index++;
                }
            }
            selectedChanged = false;
        }
        // if(currentlySelected != null && instancesSeen % 500 == 0) System.out.println(Arrays.toString(currentlySelected));
        return currentlySelected;
    }

    /**
     * Updates the feature selection method with an instance.
     *
     * @param instnc
     */
    @Override
    public void trainOnInstance (Instance instnc){
        // if the process just stated, all structures are initialized
        if (instancesSeen == 0) {
            resetStructures(instnc);
        }

        // updates the number of instances seen
        instancesSeen++;

        // updates the sliding window
        window.add(instnc);
        if (window.size() > gracePeriodOption.getValue()) {
            window.delete(0);
        }

        // weight of the instance
        double lambda = instnc.weight();
        Instance evalInstance = instnc.copy();
        int indexOfDrifted = -1; // -1 represents no drifts were flagged
        int index = -1;

        // Layers to be removed
        ArrayList<OzaBoostingLayer> toRemove = new ArrayList<>();

        // loops over all layers, adjusting the weight lambda
        // given the predictions of their internal decision stumps
        OzaBoostingLayer last = layers.get(layers.size() - 1);
        for (OzaBoostingLayer l : layers) {
            // updates l's hit rate
            boolean hit = l.correctlyClassifies(instnc);
            if (!l.isTestedForAccImprovements()) {
                l.observeHit(hit ? 1 : 0);
            }
            index++;
            // if l is not the last one
            if (l != last) {
                // updates the weights of the layers and instance following OzaBoost's method
                if (hit) {
                    l.setScms(l.getScms() + lambda);
                    lambda *= (l.getScms() + l.getSwms()) / (2 * l.getScms());
                } else {
                    l.setSwms(l.getSwms() + lambda);
                    lambda *= (l.getScms() + l.getSwms()) / (2 * l.getSwms());
                }

                // updates the stump
                l.trainOnInstance(instnc);

                // checks if l should be removed
                if (l.shouldBeRemoved()) {
                    toRemove.add(l);
                }

                // sets the attribute of this layer as missing
                evalInstance.setMissing(l.getAttributeSelected());

                // drift detected?
                if (l.isChangeDetected()) {
                    indexOfDrifted = index;
                    break;
                }

                // should this layer be removed?
                if (l.shouldBeRemoved()) {
                    if(verboseOption.isSet()) System.out.println("\t\t[removal] of layer = " + l.getAttributeSelected() + "\n");
                    toRemove.add(l);
                }
            }
        }

        // if a drift is flagged
        if (indexOfDrifted != -1) {
            // removes all layers after the drifting one
            while (layers.size() != indexOfDrifted) {
                int lastIndex = layers.size() - 1;
                OzaBoostingLayer currentLast = layers.get(layers.size() - 1);
                layers.remove(lastIndex);

                // updates the selected and unselected subsets
                selectedFeatures.remove(new Integer(currentLast.getAttributeSelected()));
                if (currentLast.getAttributeSelected() != -1){
                    unselectedFeatures.add(new Integer(currentLast.getAttributeSelected()));
                }
                selectedChanged = true;
            }

            // verbose
            if (verboseOption.isSet())
                System.out.println("[drift @ " + instancesSeen + "] \n" + this.getLayersWithMerits());

            // instantiates a new candidate layer
            layers.add(instantiateLayer());

            // resets the learner
            learner.resetLearning();
            learner.prepareForUse();

            // speeds up the learning process with the buffered instances
            for (int instIndex = 0; instIndex < window.numInstances(); instIndex++) {
                Instance filtered = filterInstance(window.get(instIndex));
                learner.trainOnInstance(filtered);
//                learner.trainOnInstance(window.get(instIndex));
            }

        } else { // no drift was flagged
            // checks if the last layer has selected a feature
            boolean wasLearning = last.getAttributeSelected() == -1;
            Instance weightedInstance = evalInstance.copy();
            weightedInstance.setWeight(lambda);
            if (lambda > 0) last.trainOnInstance(weightedInstance);
            boolean isLearning = last.getAttributeSelected() == -1;

            //verifies if the last layer has split
            if (wasLearning && !isLearning) {
                // if so, let's add a new layer
                layers.add(instantiateLayer());
                // verbose
                if (verboseOption.isSet()) System.out.println(instancesSeen + "\t" + getLayersWithMerits());

                // updates the selected and unselected subsets
                selectedFeatures.add(new Integer(last.getAttributeSelected()));
                unselectedFeatures.remove(new Integer(last.getAttributeSelected()));
                selectedChanged = true;
            }
        }


        // remove layers
        if(deleteOption.isSet()) {
            for (OzaBoostingLayer r : toRemove) {
                int iRemoved = r.getAttributeSelected();
                unselectedFeatures.add((Integer) iRemoved);
                selectedFeatures.remove((Integer) iRemoved);
                layers.remove(r);
                selectedChanged = true;
            }
        }

    }

    private void resetStructures(Instance instnc) {
        this.layers = new ArrayList<>(instnc.numAttributes() - 1);
        this.header = (InstancesHeader) instnc.dataset();
        this.setModelContext((InstancesHeader) instnc.dataset());
        this.window = new Instances(this.header, gracePeriodOption.getValue());
        this.selectedChanged = true;
        this.selectedFeatures = new ArrayList<>(instnc.numAttributes() - 1);
        this.unselectedFeatures = new ArrayList<>(instnc.numAttributes() - 1);

        // instantiates the initial layer
        layers.add(instantiateLayer());

        // all features are initially unselected
        for(int i = 0; i < header.numAttributes(); i++){
            if(i != header.classIndex()){
                unselectedFeatures.add(new Integer(i));
            }
        }
    }

    /**
     * Resets the feature selection method.
     */
    @Override
    public void resetLearning() {
        this.layers = null;
        this.selectedFeatures = null;
        this.window = null;
        this.instancesSeen = 0;
        this.header = null;
    }

    /**
     * This method describes the implementation of how to prepare this object for use.
     * All classes that extends this class have to implement <code>prepareForUseImpl</code>
     * and not <code>prepareForUse</code> since
     * <code>prepareForUse</code> calls <code>prepareForUseImpl</code>.
     *
     * @param monitor    the TaskMonitor to use
     * @param repository the ObjectRepository to use
     */
    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {
        // TODO: what do we need to put here?
    }


    /**
     * Gets the memory size of this object.
     *
     * @return the memory size of this object
     */
    @Override
    public int measureByteSize() {
        return measureByteSize(this);
    }

    /**
     * Returns a string representation of this object.
     * Used in <code>AbstractMOAObject.toString</code>
     * to give a string representation of the object.
     *
     * @param sb     the stringbuilder to add the description
     * @param indent the number of characters to indent
     */
    @Override
    public void getDescription(StringBuilder sb, int indent) {
        sb.append("A Boosting-based feature selection method for data streams.");
    }

    /**
     * Hoeffding bound.
     *
     * @param range      the range of the variable
     * @param confidence the confidence level
     * @param n          the sample size
     * @return the hoeffding bound given the parameters above
     */
    public static double computeHoeffdingBound(double range, double confidence,
                                               double n) {
        double hoeffBound = Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
        return hoeffBound;
    }

    /**
     * Chebyshev bound.
     *
     * @param range      the range of the variable
     * @param confidence the confidence level
     * @param n          the sample size
     * @return the Chebyshev bound given the parameters above
     */
    public static double computeChebyshevBound(double range, double confidence, double n) {
        double chebyshev = 1.0f / Math.sqrt((1.0f - confidence) * n);
        return chebyshev;
    }

    /**
     * Instantiates a new layer
     *
     * @return the new layer
     */
    private OzaBoostingLayer instantiateLayer() {
        OzaBoostingLayer layer = new OzaBoostingLayer();
        int gp = gracePeriodOption.getValue();
        layer.gracePeriodOption.setValue(gp);
        layer.splitCriterionOption.setValueViaCLIString(splitCriterionOption.getValueAsCLIString());
        layer.driftDetectionMethodOption.setValueViaCLIString(this.driftDetectionMethodOption.getValueAsCLIString());
        double threshold = this.selectionThresholdOption.getValue();
        if (this.selectionStrategyOption.getChosenLabel().contains("HOEFF")) {
            threshold = computeHoeffdingBound(1.0, 0.05, gracePeriodOption.getValue());
        } else if (this.selectionStrategyOption.getChosenLabel().contains("CHEBYSHEV")) {
            threshold = computeChebyshevBound(1.0, 0.05, gracePeriodOption.getValue());
        }
        layer.selectionThresholdOption.setValue(threshold);
        layer.setIndexLayer(this.layers.size() + 1);
        layer.prepareForUse();
        layer.resetLearning();
        return layer;
    }

    /**
     * Returns a string with statistics about each layer
     *
     * @return a string with statistics about each layer
     */
    private String getLayersWithMerits() {
        String str = "";
        for (OzaBoostingLayer layer : layers) {
            if (layer.getAttributeSelected() != -1) {
                str += "\n\t[" + header.attribute(layer.getAttributeSelected()).name() + " - " + layer.getMerit() + "] \t";
//                str += "\n\t[" + header.attribute(layer.getAttributeSelected()).name() + " - " + layer.getAccImprovements() + "] \t";
            } else {
                str += "\n\t[ ??? ] \t";
            }
        }
        return str;
    }

    /**
     * Flag that determines whether the selector depends on the learner or not.
     *
     * @return the flag
     */
    @Override
    public boolean dependsOnLearner() {
        return true;
    }
}
