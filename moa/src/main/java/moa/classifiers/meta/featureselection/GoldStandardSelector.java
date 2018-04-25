/*
 * Copyright (c) 2018.
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

package moa.classifiers.meta.featureselection;

import com.github.javacliparser.FlagOption;
import com.yahoo.labs.samoa.instances.Instance;
import moa.core.ObjectRepository;
import moa.tasks.TaskMonitor;

/**
 * GoldStandardSelector
 *
 * A gold-standard feature selector. It should be used as an upper bound estimate of how
 * a "perfect selector" behaves. The flag <b>resetOnFeatureChangeOption</b> determines
 * whether the classifier should be reset upon changes in the relevant subset of features.
 *
 */
public class GoldStandardSelector extends AbstractFeatureSelector {

    /**
     * A flag to determine whether the classifier should be reset upon changes in
     * the ground-truth feature subset.
     */
    public FlagOption resetOnFeatureChangeOption = new FlagOption("resetOnFeatureChange",
            'f', "Determines whether the classifier should be reset upon changes " +
            "in the gold-standard feature set.");

    /**
     * The currently relevant subset of features.
     */
    protected int currentRelevant[] = null;

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        sb.append("A gold-standard feature selector. It should be used as an upper bound " +
                "estimate of how a \"perfect selector\" behaves.");
    }


    @Override
    public int[] getSelectedFeatures() {
        return currentRelevant;
    }

    @Override
    public void trainOnInstance(Instance instnc) {
        int newRelevant[] = instnc.dataset().getIndicesRelevants();
        if(currentRelevant == null){
            currentRelevant = newRelevant;
        }else if(!eqArrays(currentRelevant, newRelevant)){
            this.currentRelevant = newRelevant;
            if(resetOnFeatureChangeOption.isSet()){
                this.learner.resetLearning();
            }
        }
    }

    @Override
    public void resetLearning() {
        this.currentRelevant = null;
    }

    @Override
    public boolean dependsOnLearner() {
        return false;
    }

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor, ObjectRepository repository) {}

    /**
     * Checks whether two <b>sorted</b> arrays <b>a</b> and <b>b</b> match.
     *
     * @param a
     * @param b
     * @return a flag
     */
    protected boolean eqArrays(int a[], int b[]){
        if(a.length != b.length) return false;
        for(int i = 0; i < a.length; i++) if(a[i] != b[i]) return false;
        return true;
    }

}
