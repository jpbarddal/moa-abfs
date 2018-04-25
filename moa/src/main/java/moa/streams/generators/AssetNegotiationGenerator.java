/*
 *    AssetNegotiationGenerator.java
 *    Copyright (C) 2017 Pontifícia Universidade Católica do Paraná, Curitiba, Brazil
 *    @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 *
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program. If not, see <http://www.gnu.org/licenses/>.
 *    
 */

package moa.streams.generators;

import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.yahoo.labs.samoa.instances.*;

import java.io.Serializable;
import java.util.*;

import moa.core.FastVector;
import moa.core.FeatureSelectionUtils;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;

/**
 * @author Jean Paul Barddal (jean.barddal@ppgia.pucpr.br)
 * @author Fabrício Enembreck (fabricio@ppgia.pucpr.br)
 * @version 2.0
 * * <p>First used in the data stream configuration in J. P. Barddal, H. M.
 * Gomes, F. Enembreck, B. Pfahringer & A. Bifet. ON DYNAMIC FEATURE WEIGHTING
 * FOR FEATURE DRIFTING DATA STREAMS. In European Conference on Machine Learning
 * and Principles and Practice of Knowledge Discovery (ECML/PKDD'16). 2016.</p>
 * <p>Originally discussed in F. Enembreck, B. C. Ávila, E. E. Scalabrin &
 * J-P. Barthès. LEARNING DRIFTING NEGOTIATIONS. In Applied Artificial
 * Intelligence: An International Journal. Volume 21, Issue 9, 2007. DOI:
 * 10.1080/08839510701526954 </p>
 */

public class AssetNegotiationGenerator
        extends AbstractOptionHandler
        implements InstanceStream, Serializable {

    /////////////
    // OPTIONS //
    /////////////

    public IntOption functionOption = new IntOption("function", 'f',
            "Classification function used, as defined in the original paper.",
            1, 1, 5);

    public FloatOption noisePercentage = new FloatOption("noise", 'n',
            "% of class noise.", 0.05, 0.0, 1.0f);

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public IntOption numIrrelevantFeaturesOption
            = new IntOption("numIrrelevantFeatures", 'F', "",
            0, 0, 1024);

    public FloatOption classRatioOption = new FloatOption("classRatio", 'R',
            "Ratio between classes.", 0.5, 0.0, 1.0);

    ///////////////
    // INTERNALS //
    ///////////////
    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    /////////////////////////
    // FEATURES DEFINITION //
    /////////////////////////
    protected transient static String colorValues[] = {"black", "blue", "cyan", "brown", "red",
            "green", "yellow", "magenta"};

    protected transient static String priceValues[] = {"veryLow", "low", "normal", "high",
            "veryHigh", "quiteHigh", "enormous", "non_salable"};

    protected transient static String paymentValues[] = {"0", "30", "60", "90", "120",
            "150", "180", "210", "240"};

    protected transient static String amountValues[] = {"veryLow", "low", "normal", "high",
            "veryHigh", "quiteHigh", "enormous", "non_ensured"};

    protected transient static String deliveryDelayValues[] = {"veryLow", "low", "normal", "high", "veryHigh"};

    protected transient static String classValues[] = {"notInterested", "interested"};

    ///////////////////////////////////
    // Class Function Implementation //
    ///////////////////////////////////
    protected interface ClassFunction {

        int determineClass(String color,
                                  String price,
                                  String payment,
                                  String amount,
                                  String deliveryDelay);

        Instance makeTrue(Instance intnc);

        int[] getRelevantFeaturesInt();

        void restart(int seed);

    }

    protected static AssetNegotiationGenerator.ClassFunction concepts[] = {
            new AssetNegotiationGenerator.ClassFunction() {
                Random r = new Random(1);

                @Override
                public int determineClass(String color,
                                          String price,
                                          String payment,
                                          String amount,
                                          String deliveryDelay) {
                    if ((price.equals("normal") && amount.equals("high")
                            || (color.equals("brown") && price.equals("veryLow")
                            && deliveryDelay.equals("high")))) {
                        return 1;
                    }
                    return 0;
                }

                @Override
                public Instance makeTrue(Instance intnc) {
                    int part = r.nextInt(2);
                    if (part == 0) {
                        intnc.setValue(1, 2); // normal price
                        intnc.setValue(3, 3); // high amount
                    } else {
                        intnc.setValue(0, 3); // brown
                        intnc.setValue(1, 0); // verylow price
                        intnc.setValue(4, 4); // high delivery delay
                    }
                    intnc.setClassValue(1);
                    return intnc;
                }

                @Override
                public int[] getRelevantFeaturesInt() {
                    return new int[]{0, 1, 3, 4}; // price, amount, color, deliveryDelay
                }


                @Override
                public void restart(int seed) {
                    r = new Random(seed);
                }
            },
            new AssetNegotiationGenerator.ClassFunction() {
                Random r = new Random(1);

                @Override
                public int determineClass(String color,
                                          String price,
                                          String payment,
                                          String amount,
                                          String deliveryDelay) {
                    if (price.equals("high") && amount.equals("veryHigh")
                            && deliveryDelay.equals("high")) {
                        return 1;
                    }
                    return 0;
                }

                @Override
                public Instance makeTrue(Instance intnc) {
                    intnc.setValue(1, 3); // high price
                    intnc.setValue(3, 4); //very high amount
                    intnc.setValue(4, 3); // high delivery delay
                    intnc.setClassValue(1);
                    return intnc;
                }

                @Override
                public int[] getRelevantFeaturesInt() {
                    return new int[]{1, 3, 4}; // price, amount, deliveryDelay
                }


                @Override
                public void restart(int seed) {
                    r = new Random(seed);
                }
            },
            new AssetNegotiationGenerator.ClassFunction() {
                Random r = new Random(1);

                @Override
                public int determineClass(String color,
                                          String price,
                                          String payment,
                                          String amount,
                                          String deliveryDelay) {
                    if ((price.equals("veryLow")
                            && payment.equals("0") && amount.equals("high"))
                            || (color.equals("red") && price.equals("low")
                            && payment.equals("30"))) {
                        return 1;
                    }
                    return 0;
                }

                @Override
                public Instance makeTrue(Instance intnc) {
                    int part = r.nextInt(2);
                    if (part == 0) {
                        intnc.setValue(1, 0); // very low price
                        intnc.setValue(2, 0); // 0 payment
                        intnc.setValue(3, 3); // high amount
                    } else {
                        intnc.setValue(0, 4); // red color
                        intnc.setValue(1, 1); // low price
                        intnc.setValue(2, 1); // 30 payment
                    }
                    intnc.setClassValue(1);
                    return intnc;
                }

                @Override
                public int[] getRelevantFeaturesInt() {
                    return new int[]{0, 1, 2, 3}; // price, payment, amount, color
                }


                @Override
                public void restart(int seed) {
                    r = new Random(seed);
                }
            },
            new AssetNegotiationGenerator.ClassFunction() {
                Random r = new Random(1);

                @Override
                public int determineClass(String color,
                                          String price,
                                          String payment,
                                          String amount,
                                          String deliveryDelay) {
                    if ((color.equals("black")
                            && payment.equals("90")
                            && deliveryDelay.equals("veryLow"))
                            || (color.equals("magenta")
                            && price.equals("high")
                            && deliveryDelay.equals("veryLow"))) {
                        return 1;
                    }
                    return 0;
                }

                @Override
                public Instance makeTrue(Instance intnc) {
                    int part = r.nextInt(2);
                    if (part == 0) {
                        intnc.setValue(0, 0); // black color
                        intnc.setValue(2, 3); // 90 payment
                        intnc.setValue(4, 0); //very delivery low delay
                    } else {
                        intnc.setValue(0, 7); // magenta color
                        intnc.setValue(1, 3); // high price
                        intnc.setValue(4, 0); // very delivery low delay
                    }
                    intnc.setClassValue(1);
                    return intnc;
                }

                @Override
                public int[] getRelevantFeaturesInt() {
                    return new int[]{0, 1, 2, 4}; //color, payment, deliveryDelay, price
                }


                @Override
                public void restart(int seed) {
                    r = new Random(seed);
                }
            },
            new AssetNegotiationGenerator.ClassFunction() {
                Random r = new Random(1);

                @Override
                public int determineClass(String color,
                                          String price,
                                          String payment,
                                          String amount,
                                          String deliveryDelay) {
                    if ((color.equals("blue")
                            && payment.equals("60")
                            && amount.equals("low")
                            && deliveryDelay.equals("normal"))
                            || (color.equals("cyan")
                            && amount.equals("low")
                            && deliveryDelay.equals("normal"))) {
                        return 1;
                    }
                    return 0;
                }

                @Override
                public Instance makeTrue(Instance intnc) {
                    int part = r.nextInt(2);
                    if (part == 0) {
                        intnc.setValue(0, 1); // blue color
                        intnc.setValue(2, 2); // 60 payment
                        intnc.setValue(3, 1); // low amount
                        intnc.setValue(4, 2); // normal delay
                    } else {
                        intnc.setValue(0, 2); // cyan color
                        intnc.setValue(3, 1); // low amount
                        intnc.setValue(4, 2); //normal delivery delay
                    }
                    intnc.setClassValue(1);
                    return intnc;
                }

                @Override
                public int[] getRelevantFeaturesInt() {
                    return new int[]{0, 2, 3, 4}; //color, payment, amount, deliveryDelay
                }

                @Override
                public void restart(int seed) {
                    r = new Random(seed);
                }
            }
    };

    //////////////////////////////
    // INTERFACE IMPLEMENTATION //
    //////////////////////////////
    protected transient AssetNegotiationGenerator.ClassFunction classFunction;

    @Override
    protected void prepareForUseImpl(TaskMonitor tm, ObjectRepository or) {
        restart();
    }

    @Override
    public void getDescription(StringBuilder sb, int i) {}

    @Override
    public InstancesHeader getHeader() {
        return streamHeader;
    }

    @Override
    public long estimatedRemainingInstances() {
        return Integer.MAX_VALUE;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }

    @Override
    public InstanceExample nextInstance() {
        if (classFunction == null) {
            classFunction = concepts[this.functionOption.getValue() - 1];
            classFunction.restart(this.instanceRandomSeedOption.getValue());
        }

        Instance instnc = null;

        boolean nextClassShouldBeZero = this.instanceRandom.nextDouble() > classRatioOption.getValue();
        boolean classFound = false;
        while (!classFound) {
            //randomize indexes for new instance
            int indexColor = this.instanceRandom.nextInt(colorValues.length);
            int indexPrice = this.instanceRandom.nextInt(priceValues.length);
            int indexPayment = this.instanceRandom.nextInt(paymentValues.length);
            int indexAmount = this.instanceRandom.nextInt(amountValues.length);
            int indexDelivery = this.instanceRandom.nextInt(deliveryDelayValues.length);

            //retrieve values
            String color = colorValues[indexColor];
            String price = priceValues[indexPrice];
            String payment = paymentValues[indexPayment];
            String amount = amountValues[indexAmount];
            String delivery = deliveryDelayValues[indexDelivery];

            int classValue = classFunction.
                    determineClass(color, price, payment, amount, delivery);

            instnc = new DenseInstance(streamHeader.numAttributes());
            //set values
            instnc.setDataset(this.getHeader());
            instnc.setValue(0, indexColor);
            instnc.setValue(1, indexPrice);
            instnc.setValue(2, indexPayment);
            instnc.setValue(3, indexAmount);
            instnc.setValue(4, indexDelivery);

            for (int i = 0; i < numIrrelevantFeaturesOption.getValue(); i++) {
                instnc.setValue(i + 5, instanceRandom.nextDouble());
            }

            if (classValue == 0 && !nextClassShouldBeZero) {
                instnc = classFunction.makeTrue(instnc);
                classValue = 1;
                classFound = true;
            } else if (classValue == 0 && nextClassShouldBeZero) {
                classFound = true;
            } else if (classValue == 1 && !nextClassShouldBeZero) {
                classFound = true;
            }
            instnc.setClassValue(classValue);

        }
        //add noise
        int newClassValue = addNoise((int) instnc.classValue());
        instnc.setClassValue(newClassValue);
        instnc.setDataset(this.streamHeader);
        return new InstanceExample(instnc);
    }

    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public void restart() {
        // generate header
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        classFunction = concepts[this.functionOption.getValue() - 1];
        classFunction.restart(this.instanceRandomSeedOption.getValue());

        ArrayList<String> attributesNames = new ArrayList<>();
        FastVector attributes = new FastVector();
        attributes.addElement(new Attribute("color",
                Arrays.asList(colorValues)));
        attributesNames.add("color");
        attributes.addElement(new Attribute("price",
                Arrays.asList(priceValues)));
        attributesNames.add("price");
        attributes.addElement(new Attribute("payment",
                Arrays.asList(paymentValues)));
        attributesNames.add("payment");
        attributes.addElement(new Attribute("amount",
                Arrays.asList(amountValues)));
        attributesNames.add("amount");
        attributes.addElement(new Attribute("deliveryDelay",
                Arrays.asList(deliveryDelayValues)));
        attributesNames.add("deliveryDelay");

        for (int i = 0; i < numIrrelevantFeaturesOption.getValue(); i++) {
            attributes.addElement(new Attribute("irrel" + i));
            attributesNames.add(("irrel" + i));
        }


        FastVector classLabels = new FastVector();
        for (int i = 0; i < classValues.length; i++) {
            classLabels.addElement(classValues[i]);
        }

        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.streamHeader.setIndicesRelevants(getRelevantFeaturesInt());
        classFunction.restart(this.instanceRandomSeedOption.getValue());

        //Outputs all relevant attributes' names
        System.out.print("relevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if(FeatureSelectionUtils.contains(i, classFunction.getRelevantFeaturesInt())){
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        // irrelevant
        System.out.print("irrelevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if(i != streamHeader.classIndex() &&
                    !FeatureSelectionUtils.contains(i, classFunction.getRelevantFeaturesInt())){
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \n");

    }

    //////////////////////
    // AUXILIAR METHODS //
    //////////////////////
    int addNoise(int classObtained) {
        if (this.instanceRandom.nextFloat() <= this.noisePercentage.getValue()) {
            classObtained = classObtained == 0 ? 1 : 0;
        }
        return classObtained;
    }

    public TreeSet<String> getRelevantFeatures() {
        TreeSet<String> relevants = new TreeSet<>();
        for (int i = 0; i < streamHeader.numAttributes(); i++) {
            Attribute att = streamHeader.attribute(i);
            if (FeatureSelectionUtils.contains(i, classFunction.getRelevantFeaturesInt())) {
                relevants.add(att.name());
            }
        }
        return relevants;
    }

    public TreeSet<String> getIrrelevantFeatures() {
        TreeSet<String> irrelevants = new TreeSet<>();
        int irrels[] = this.getIrrelevantFeaturesInt();
        for (int i = 0; i < irrels.length; i++) {
            Attribute att = streamHeader.attribute(irrels[i]);
            irrelevants.add(att.name());
        }
        return irrelevants;
    }

    public TreeSet<String> getRedundantFeatures() {
        return new TreeSet<>();
    }

    public HashMap<String, String> getRedundantToFeatures() {
        return new HashMap<>();
    }


    /**
     * Method that results an array with the indices
     * of the ground-truth <b>relevant</b> features.
     *
     * @return an array with the indices of relevant features
     */
    public int[] getRelevantFeaturesInt() {
        return concepts[this.functionOption.getValue() - 1].getRelevantFeaturesInt();
    }

    /**
     * Method that results an array with the indices
     * of the ground-truth <b>irrelevant</b> features.
     *
     * @return an array with the indices of irrelevant features
     */
    public int[] getIrrelevantFeaturesInt() {
        int rel[] = this.getRelevantFeaturesInt();
        int irrel[] = new int[streamHeader.numAttributes() - rel.length - 1];
        int index = 0;
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            if(streamHeader.classIndex() != i &&
                    !FeatureSelectionUtils.contains(i, rel)){
                irrel[index] = i;
                index++;
            }
        }
        return irrel;
    }
}