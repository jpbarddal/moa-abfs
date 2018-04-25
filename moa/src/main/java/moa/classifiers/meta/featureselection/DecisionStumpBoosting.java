package moa.classifiers.meta.featureselection;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.TreeSet;

import com.github.javacliparser.FloatOption;
import moa.classifiers.core.AttributeSplitSuggestion;
import moa.classifiers.core.attributeclassobservers.AttributeClassObserver;
import moa.classifiers.core.splitcriteria.SplitCriterion;
import moa.classifiers.trees.DecisionStump;
import moa.core.Utils;

/**
 *
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @version 1.0
 */
public class DecisionStumpBoosting extends DecisionStump {

    public FloatOption minMeritForSplitOption = new FloatOption("minMeritForSplit", 'M',
            "Threshold for minimum merit.", 1e-10, 0.0, 1.0);

    TreeSet<FeatureScore> attsScores = new TreeSet<FeatureScore>();

    public DecisionStumpBoosting() {
        this.attsScores = new TreeSet<>();
    }

    public AttributeSplitSuggestion getBestSplit() {
        return bestSplit;
    }

    public TreeSet<FeatureScore> getAttsScores() {
        return attsScores;
    }

    @Override
    protected AttributeSplitSuggestion findBestSplit(SplitCriterion criterion) {
        attsScores.clear();
        AttributeSplitSuggestion bestFound = null;
        double bestMerit = Double.NEGATIVE_INFINITY;
        int bestIndex = -1;
        double[] preSplitDist = this.observedClassDistribution.getArrayCopy();
        for (int i = 0; i < this.attributeObservers.size(); i++) {
            AttributeClassObserver obs = this.attributeObservers.get(i);
            if (obs != null) {
                AttributeSplitSuggestion suggestion = obs.getBestEvaluatedSplitSuggestion(criterion,
                        preSplitDist, i, this.binarySplitsOption.isSet());
                if (suggestion != null) {
                    attsScores.add(new FeatureScore(i, suggestion.merit));
                    if (suggestion.merit > bestMerit &&
                            suggestion.merit > minMeritForSplitOption.getValue()) {
                        bestMerit = suggestion.merit;
                        bestFound = suggestion;
                        bestIndex = i;
                    }
                }
            }
        }
        if (bestMerit == Double.NEGATIVE_INFINITY) {
            attsScores.clear();
        } else if (bestMerit == 0.0) {
            bestFound = null;
        } else {
            // filters the attsScores so they maintain only the most 
            // "similar" values, so these are potential redundant features
            ArrayList<FeatureScore> toRemove = new ArrayList<>();
            for (FeatureScore s : attsScores) {
                if (Math.abs(s.score - bestMerit) > 0.1 || s.attIndex == bestIndex) {
                    toRemove.add(s);
                }
            }
            attsScores.removeAll(toRemove);
        }

        return bestFound;
    }

    public static double computeHoeffdingBound(double range, double confidence,
            double n) {
        return Math.sqrt(((range * range) * Math.log(1.0 / confidence))
                / (2.0 * n));
    }

    public class FeatureScore implements Comparable<FeatureScore>, Serializable {

        int attIndex;
        double score;

        public FeatureScore(int attIndex, double score) {
            this.attIndex = attIndex;
            this.score = score;
        }

        public int getAttIndex() {
            return attIndex;
        }

        public double getScore() {
            return score;
        }

        @Override
        public int compareTo(FeatureScore o) {
            if (this.score < o.score) {
                return -1;
            } else if (this.score == o.score) {
                if (this.attIndex < o.attIndex) {
                    return -1;
                } else if (this.attIndex == o.attIndex) {
                    return 0;
                } else {
                    return +1;
                }
            } else {
                return +1;
            }
        }

    }

}
