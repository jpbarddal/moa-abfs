
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

package moa.classifiers.meta.featureselection;

import com.yahoo.labs.samoa.instances.FilteredSparseInstance;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.Classifier;
import moa.options.AbstractOptionHandler;

public abstract class AbstractFeatureSelector extends AbstractOptionHandler
        implements FeatureSelector {

    /**
     * A link to the learner, that may or not be used depending on
     * the feature selection procedure adopted.
     */
    protected Classifier learner;

    /**
     * The header of the original instances.
     */
    protected InstancesHeader header;

    /**
     * Sets the learner.
     * @param learner
     */
    public void setLearner(Classifier learner) {
        this.learner = learner;
    }

    /**
     * Filters an instance with the currently selected features.
     *
     * @param instnc the original instance
     * @return the instance with only the selected features
     */
    @Override
    public Instance filterInstance(Instance instnc) {
        // retrieves the selected features
        int selected[] = this.getSelectedFeatures();
        Instance filtered;
        if(selected != null && selected.length > 0 && selected.length < instnc.numAttributes() - 1){
            int numAttributes = instnc.numAttributes();

            // copies all values including the class
            int indices[] = new int[selected.length + 1];
            double values[] = new double[selected.length + 1];
            for (int i = 0; i < selected.length; i++) {
                indices[i] = selected[i];
                values[i] = instnc.value(selected[i]);
            }
            //adds the class index and value
            indices[indices.length - 1] = instnc.classIndex();
            values[indices.length - 1] = instnc.classValue();

            filtered = new FilteredSparseInstance(1.0, values, indices, numAttributes);
            filtered.setDataset(instnc.dataset());
        }else{
            filtered = instnc;
        }
        return filtered;
    }

    /**
     * Returns the number of features currently selected by the method.
     *
     * @return the number of features selected
     */
    @Override
    public int numFeaturesSelected() {
        int s[] = this.getSelectedFeatures();
        if(s == null) return 0;
        return s.length;
    }

    /**
     * Sets the header h
     *
     * @param h the header
     */
    @Override
    public void setModelContext(InstancesHeader h) {
        this.header = h;
    }

}
