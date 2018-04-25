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

import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import moa.streams.ConceptDriftStreamDetailingFeatures;
import moa.streams.ExampleStream;
import moa.streams.IrrelevantFeatureAppenderStream;
import moa.streams.generators.*;

public class FactoryConceptDriftedStreams {

    private final int NUM_IRRELEVANT_FEATURES; // 20
    private final int STREAM_LENGTH; // 1000000
    private final int NUM_CONCEPTS; // 10
    private final int GRADUAL_DRIFT_LENGTH; // 10000

    public FactoryConceptDriftedStreams(int NUM_IRRELEVANT_FEATURES,
            int STREAM_LENGTH,
            int NUM_CONCEPTS,
            int GRADUAL_DRIFT_LENGTH) {
        this.NUM_IRRELEVANT_FEATURES = NUM_IRRELEVANT_FEATURES;
        this.STREAM_LENGTH = STREAM_LENGTH;
        this.NUM_CONCEPTS = NUM_CONCEPTS;
        this.GRADUAL_DRIFT_LENGTH = GRADUAL_DRIFT_LENGTH;
    }

    public HashMap<String, ExampleStream> instantiateAll() {
        HashMap<String,ExampleStream> all = new HashMap<>();
        all.putAll(instantiateASSETS());
        all.putAll(instantiateAGRAWAL());
        all.putAll(instantiateBG1());
        all.putAll(instantiateBG2());
        all.putAll(instantiateBG3());
        all.putAll(instantiateRTG());
        all.putAll(instantiateSEA());
        return all;
    }

    public HashMap<String, ExampleStream> instantiateAGRAWAL() {
        int concepts[] = new int[]{1, 2, 3, 4};
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            AgrawalGenerator agr = new AgrawalGenerator();
            agr.balanceClassesOption.set();
            agr.functionOption.setValue(concepts[i % concepts.length]);
            agr.prepareForUse();

            IrrelevantFeatureAppenderStream appended = new IrrelevantFeatureAppenderStream();
            appended.numNumericFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES / 2);
            appended.numCategoricalFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES / 2);
            appended.streamOption.setCurrentObject(agr);
            appended.prepareForUse();
//            System.out.println("---" + Arrays.toString(appended.getHeader().getIndicesRelevants()));

            substreams.add((ExampleStream) appended.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("AGR(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("AGR(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateASSETS() {
        int concepts[] = new int[]{1, 2, 3, 4};
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            AssetNegotiationGenerator an
                    = new AssetNegotiationGenerator();
            an.functionOption.setValue(concepts[i % concepts.length]);
            an.numIrrelevantFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES);
            an.prepareForUse();

            IrrelevantFeatureAppenderStream appended = new IrrelevantFeatureAppenderStream();
            appended.numNumericFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES / 2);
            appended.numCategoricalFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES / 2);
            appended.streamOption.setCurrentObject(an);
            appended.prepareForUse();

            substreams.add((ExampleStream) appended.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("AN(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("AN(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateBG1() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        String relevants[] = new String[]{"0;1;2", "3;4;5", "1;3;5", "0;1;3"};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            BG bg = new BG();
            bg.numFeaturesOption.setValue(3 + NUM_IRRELEVANT_FEATURES);
            bg.balanceClassesOption.set();
            bg.relevantFeaturesOption.setValue(relevants[i % relevants.length]);
            bg.prepareForUse();
            substreams.add((ExampleStream) bg.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("BG1(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("BG1(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateBG2() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        String relevants[] = new String[]{"0;1;2", "3;4;5", "1;3;5", "0;1;3"};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            BG2 bg = new BG2();
            bg.numFeaturesOption.setValue(3 + NUM_IRRELEVANT_FEATURES);
            bg.balanceClassesOption.set();
            bg.relevantFeaturesOption.setValue(relevants[i % relevants.length]);
            bg.prepareForUse();
            substreams.add((ExampleStream) bg.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("BG2(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("BG2(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateBG3() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        int seeds[] = new int[]{78613, 897131, 2311};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            BG3 bg = new BG3();
            bg.instanceRandomSeedOption.setValue(seeds[i % seeds.length]);
            bg.balanceClassesOption.set();
            bg.numIrrelevantFeaturesOption.setValue(NUM_IRRELEVANT_FEATURES);
            bg.prepareForUse();
            substreams.add((ExampleStream) bg.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("BG3(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("BG3(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateLED() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        int numDriftingFeatures[] = new int[]{0, 4, 2};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            LEDGeneratorDrift led = new LEDGeneratorDrift();
            led.numberAttributesDriftOption.setValue(numDriftingFeatures[i % numDriftingFeatures.length]);
            led.prepareForUse();
            substreams.add((ExampleStream) led.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("LED(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("LED(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateRTG() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        int seeds[] = new int[]{1, 2, 3, 4};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            RTGFD rtg = new RTGFD();
            rtg.treeRandomSeedOption.setValue(seeds[i % seeds.length]);
            rtg.instanceRandomSeedOption.setValue(seeds[i % seeds.length]);
            rtg.numNominalsOption.setValue(3 + (NUM_IRRELEVANT_FEATURES / 2));
            rtg.numNumericsOption.setValue(3 + (NUM_IRRELEVANT_FEATURES / 2));
            rtg.numClassesOption.setValue(2);
            rtg.numIrrelOption.setValue(NUM_IRRELEVANT_FEATURES);
            rtg.prepareForUse();
            substreams.add((ExampleStream) rtg.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("RTG(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("RTG(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public HashMap<String, ExampleStream> instantiateSEA() {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        int seeds[] = new int[]{1327123871, 1231245, 612372178, 12372178};
        int concepts[] = new int[]{1, 4, 2, 3};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            SEAFD sea = new SEAFD();
            sea.numRandomAttsOption.setValue(NUM_IRRELEVANT_FEATURES);
            sea.instanceRandomSeedOption.setValue(seeds[i % seeds.length]);
            sea.functionOption.setValue(concepts[i % concepts.length]);
            sea.balanceClassesOption.set();
            sea.prepareForUse();
            substreams.add((ExampleStream) sea.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("SEA(A)", createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("SEA(G)", createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

    public static ExampleStream createDriftedStream(
            LinkedList<ExampleStream> concepts,
            int driftLength, int streamLength) {
        int numDrifts = concepts.size() - 1;
        int position = -1;
        if (numDrifts == 0) {
            ExampleStream c = concepts.getFirst();
            return c;
        } else {
            position = (int) Math.floor((double) streamLength / (numDrifts + 1));
        }

        // instantiates the substreams, 
        // each being a ConceptDriftStream with details
        // also, the priori concept is also set
        LinkedList<ExampleStream> parts = new LinkedList<>();
        for (int i = 0; i < numDrifts; i++) {
            ConceptDriftStreamDetailingFeatures c
                    = new ConceptDriftStreamDetailingFeatures();
            c.widthOption.setValue(driftLength);
            c.positionOption.setValue(position);
            c.streamOption.setCurrentObject(concepts.get(i)/*.copy()*/);
//            c.prepareForUse();
            parts.add((ExampleStream) c/*.copy()*/);
        }

        // now, we set the driftstream accordingly
        for (int i = 0; i < parts.size(); i++) {
            ConceptDriftStreamDetailingFeatures current
                    = (ConceptDriftStreamDetailingFeatures) parts.get(i);
            ExampleStream next = null;
            if (i + 1 >= parts.size()) {
                next = concepts.getLast();
            } else {
                next = parts.get(i + 1);
            }
            current.driftstreamOption.setCurrentObject(next);
        }

        // just to be sure, let's prepare all the inner streams
        for (ExampleStream str : parts) {
            ((ConceptDriftStreamDetailingFeatures) str).prepareForUse();
        }

        return parts.getFirst();
    }

    public HashMap<String, ExampleStream> instantiateAllHyperplaneReg(){
        HashMap<String, ExampleStream> hyps = new HashMap<>();
        hyps.putAll(instantiateHyperplaneRegDistance());
        hyps.putAll(instantiateHyperplaneRegSqDistance());
        hyps.putAll(instantiateHyperplaneRegCubicDistance());
        return hyps;
    }

    // type = "Distance", "SquareDistance", "CubicDistance"

    public HashMap<String, ExampleStream> instantiateHyperplaneRegDistance(){
        return instantiateHyperplaneReg("Distance");
    }

    public HashMap<String, ExampleStream> instantiateHyperplaneRegSqDistance(){
        return instantiateHyperplaneReg("SquareDistance");
    }

    public HashMap<String, ExampleStream> instantiateHyperplaneRegCubicDistance(){
        return instantiateHyperplaneReg("CubicDistance");
    }

    public HashMap<String, ExampleStream> instantiateHyperplaneReg(String type) {
        LinkedList<ExampleStream> substreams = new LinkedList<>();
        int seeds[] = new int[]{1327123871, 1231245, 612372178, 12372178};
        for (int i = 0; i < NUM_CONCEPTS; i++) {
            HyperplaneGeneratorReg hyp = new HyperplaneGeneratorReg();
            hyp.numAttsOption.setValue(10);
            hyp.numDriftAttsOption.setValue(0);
            hyp.targetValueOption.setChosenLabel(type);
            hyp.instanceRandomSeedOption.setValue(seeds[i]);
            hyp.prepareForUse();
            substreams.add((ExampleStream) hyp.copy());
        }

        // creates final hash
        HashMap<String, ExampleStream> ret
                = new HashMap<String, ExampleStream>();
        ret.put("HYP" + type + "(A)" , createDriftedStream(substreams,
                1,
                STREAM_LENGTH));
        ret.put("HYP" + type + "(G)" , createDriftedStream(substreams,
                GRADUAL_DRIFT_LENGTH,
                STREAM_LENGTH));
        return ret;
    }

}
