
/*
 * Copyright (c) 2017.
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Heitor Murilo Gomes (heitor.gomes@telecom-paristech.fr)
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

package moa.classifiers.meta.featureselection.;

import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.Classifier;

import java.io.Serializable;

/**
 *
 * Feature Selector Interface
 *
 * This interface provides a common protocol for feature selection algorithms in MOA.
 * In practice, a feature selection algorithm should return, at any given timestamp,
 * an array with the indices of the selected features.
 *
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Heitor Murilo Gomes (heitor.gomes@telecom-paristech.fr)
 * @version 1.0
 */
public interface FeatureSelector {

    /**
     * Method responsible for returning the selected subset of features.
     *
     * @return the indices of the selected features
     */
    public int[] getSelectedFeatures();

    /**
     * Filters an instance with the currently selected features.
     *
     * @param instnc the original instance
     * @return the instance with only the selected features.
     */
    public Instance filterInstance(Instance instnc);

    /**
     * Updates the feature selection method with an instance.
     *
     * @param instnc
     */
    public void trainOnInstance(Instance instnc);


    /**
     * Returns the number of features currently selected by the method.
     *
     * @return the number of features selected
     */
    public int numFeaturesSelected();

    /**
     * Resets the feature selection method.
     */
    public void resetLearning();

    /**
     * Prepares the feature selector for use.
     */
    public void prepareForUse();

    /**
     * Sets the header h
     *
     * @param h the header
     */
    public void setModelContext(InstancesHeader h);

    /**
     * Flag that determines whether the selector depends on the learner or not.
     *
     * @return the flag
     */
    public boolean dependsOnLearner();

}
