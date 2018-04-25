
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

package moa.classifiers.featureselection;

import com.github.javacliparser.FlagOption;
import moa.classifiers.MultiClassClassifier;
import moa.options.ClassOption;
import com.github.javacliparser.FloatOption;
import com.yahoo.labs.samoa.instances.Instance;
import com.yahoo.labs.samoa.instances.InstancesHeader;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.FeatureSelectionUtils;
import moa.core.Measurement;

import java.util.Arrays;


/**
 * FeatureSelectionClassifier
 *
 * This abstract class determines the behavior of classifiers
 * that will work associated with a feature selection process.
 */
public class FeatureSelectionClassifier extends AbstractClassifier implements MultiClassClassifier {

    /**
     * The classifier.
     */
    protected Classifier learner;

    /**
     * The feature selection process.
     */
    protected AbstractFeatureSelector selector;

    /**
     * A pointer to the last instance seen during the training step.
     */
    private Instance lastInstance;

    /**
     * A flag on whether this object is being used for stability computation.
     * It will be set by the <p>EvaluateFeatureSelectionStability</p> task to true,
     * otherwise, it should use the learner.
     */
    private boolean computingStability = false;

    /**
     * Options to set up the items above.
     */
    public moa.options.ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");
    public ClassOption selectorOption = new ClassOption("selector", 's',
            "Selector used to perform feature selection.", AbstractFeatureSelector.class, "newfeatureselection.BoostingSelector");
    public FloatOption gammaOption = new FloatOption("gamma", 'g',
            "Weighting factor for Selection Accuracy computation.", 0.7f, 0.0f, 1.0f);
    public FlagOption doNotComputeSelectionMetricsOption =
            new FlagOption("doNotComputeSelectionMetrics", 'm',
                    "Determines whether computation of feature selection metrics should be skipped.");

    /**
     * Returns the prediction from the learner.
     *
     * @param inst the instance used for prediction
     * @return the predictions from the learner
     */
    @Override
    public double[] getVotesForInstance(Instance inst) {
        // first, filters the instance so only the selected features are used
        Instance filtered = selector.filterInstance(inst);
        // and then trains the learner
        return learner.getVotesForInstance(filtered);
    }

    /**
     * Gets whether this learner needs a random seed.
     * Examples of methods that needs a random seed are bagging and boosting.
     *
     * @return true if the learner needs a random seed.
     */
    @Override
    public boolean isRandomizable() {
        return false;
    }


    @Override
    public void setModelContext(InstancesHeader ih) {
        super.setModelContext(ih);
        this.learner.setModelContext(ih);
        this.selector.setModelContext(ih);
    }

    /**
     * Returns the measurements from both the classifier and also from the feature selector.
     *
     * @return the measurements
     */
    @Override
    public Measurement[] getModelMeasurements() {
        // the learner metrics
        Measurement mClassifier[] = learner.getModelMeasurements();
        Measurement mSelector[] = !doNotComputeSelectionMetricsOption.isSet() ?
                getSelectionMetrics() : new Measurement[0];

        Measurement[] result = Arrays.copyOf(mClassifier, mClassifier.length + mSelector.length);
        System.arraycopy(mSelector, 0, result, mClassifier.length, mSelector.length);
        return result;
    }

    /**
     * Resets this classifier. It must be similar to
     * starting a new classifier from scratch. <br><br>
     * <p>
     * The reason for ...Impl methods: ease programmer burden by not requiring
     * them to remember calls to super in overridden methods.
     * Note that this will produce compiler errors if not overridden.
     */
    @Override
    public void resetLearningImpl() {
        this.learner = (Classifier) getPreparedClassOption(baseLearnerOption);
        this.selector = (AbstractFeatureSelector) getPreparedClassOption(selectorOption);

        // sets the learner pointer
        this.selector.learner = learner;

        this.learner.prepareForUse();
        this.selector.prepareForUse();

        this.learner.resetLearning();
        this.selector.resetLearning();
    }

    /**
     * Updates the feature selection model and also the learner with the selected features.
     *
     * @param inst the instance to be used for training
     */
    @Override
    public void trainOnInstanceImpl(Instance inst) {
        // stores this instance for SA computation
        lastInstance = inst;

        // train the feature selection method
        selector.trainOnInstance(inst);

        // train the learner
        if(!this.computingStability || this.selector.dependsOnLearner()) {
            Instance filtered = selector.filterInstance(inst);
            learner.trainOnInstance(filtered);
        }
    }

    /**
     * Gets the current measurements of this classifier.<br><br>
     * <p>
     * The reason for ...Impl methods: ease programmer burden by not requiring
     * them to remember calls to super in overridden methods.
     * Note that this will produce compiler errors if not overridden.
     *
     * @return an array of measurements to be used in evaluation tasks
     */
    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return this.getModelMeasurements();
    }

    /**
     * Returns a string representation of the model.
     *
     * @param out    the stringbuilder to add the description
     * @param indent the number of characters to indent
     */
    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        out.append("A wrapper that combines a learner with a feature selection method.");
    }

    /**
     * Returns the selection metrics, including the Selection Accuracy,
     * Recall of Relevant Features and the Complexity penalty measurements.
     *
     * @return the selection metrics
     */
    private Measurement[] getSelectionMetrics() {
        double m[] = computeSelectionAccuracyMetrics();
        return new Measurement[]{
                new Measurement("# of features selected", this.selector.numFeaturesSelected()),
                new Measurement("pct of features selected (%)",
                        100.0 * this.selector.numFeaturesSelected() / (lastInstance.numAttributes() - 1)),
                new Measurement("Selection Accuracy (SA)", m[0]),
                new Measurement("Recall of Relevant Features (RRF)", m[1]),
                new Measurement("Complement of Complexity Penalty (CCP)", m[2])
        };
    }

    /**
     * Calculates the Selection Accuracy (SA) metrics according to
     * <p>L. C. Molina, L. Belanche and A. Nebot, "Feature selection algorithms: a survey
     * and experimental evaluation," 2002 IEEE International Conference on Data Mining,
     * 2002. Proceedings., 2002, pp. 306-313. doi: 10.1109/ICDM.2002.1183917</p>
     *
     * @return an array with 3 values: selection accuracy, the recall of relevant
     * features, and a penalty for selecting extraneous features.
     */
    private double[] computeSelectionAccuracyMetrics() {

        // gets the selected features
        int selected[] = selector.getSelectedFeatures();

        if (selected == null ||
                selected.length == 0 ||
                lastInstance.dataset().getIndicesRelevants() == null)
            return new double[]{Double.NaN, Double.NaN, Double.NaN};

        // gets the ground-truth features
        int relevant[] = lastInstance.dataset().getIndicesRelevants();
        int irrelevant[] = lastInstance.dataset().getIndicesIrrelevants();

        // SA = \gamma(k / K) + (1 - \gamma)(1 - (p / (P - K)))
        // where
        // k is the # of relevant features SELECTED
        // K is total number of RELEVANT inputs
        // p is the number of extraneous (IRRELEVANT + REDUNDANT) features SELECTED
        // P is total number of inputs (RELEVANT + IRRELEVANT + REDUNDANT)
        // I = (P - K) is the number of irrelevant features

        long k = FeatureSelectionUtils.intersection(selected, relevant);
        long K = relevant.length;
        long p = FeatureSelectionUtils.intersection(selected, irrelevant);
        long I = irrelevant.length;

        // components
        double rrf = k / (double) K;
        double ccp = 1.0 - (p / (double) I);

        // SA computation
        double gamma = gammaOption.getValue();
        double sa = gamma * rrf + (1.0 - gamma) * ccp;

        // format: SA, Recall Relevant, Complement of Complexity Penalty
        return new double[]{sa, rrf, ccp};
    }

    /**
     * Sets the flag for stability computation.
     * If this flag is set, the learner will be updated only if <p>dependsOnLearner=True</p>
     */
    public void setComputingStability() {
        this.computingStability = true;
    }


    /**
     * Returns the selected features from the internal selector.
     *
     * @return the indices of the selected features
     */
    public int[] getSelectedFeatures(){
        return this.selector.getSelectedFeatures();
    }
}
