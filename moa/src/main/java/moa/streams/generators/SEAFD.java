package moa.streams.generators;

import com.github.javacliparser.FlagOption;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import com.yahoo.labs.samoa.instances.*;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.TreeSet;
import moa.core.InstanceExample;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import org.apache.commons.collections15.map.FastHashMap;
import weka.core.FastVector;

/**
 *
 * @author Jean Paul Barddal
 */
public class SEAFD
        extends AbstractOptionHandler
        implements InstanceStream {
    
    @Override
    public String getPurposeString() {
        return "Generates SEA concepts functions.";
    }
    
    public IntOption functionOption = new IntOption("function", 'f',
            "Classification function used, as defined in the original paper.",
            1, 1, 4);
    
    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);
    
    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");
    
    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 10, 0, 100);
    
    public IntOption numRandomAttsOption
            = new IntOption("numRandomAtts", 'R', "", 0, 0, 1024);
    
    public IntOption numRedundantFeaturesOption
            = new IntOption("numRedundantFeatures", '-',
                    "# of redundant features", 0, 0, 1000);

    public IntOption numCosineNonLinearRedundantFeaturesOption
            = new IntOption("numCosineNonLinearRedundantFeatures", '_',
                    "# of cosine non linear redundant features", 0, 0, 1000);

    public IntOption numRBFlikeNonLinearRedundantFeaturesOption
            = new IntOption("numRBFLikeNonLinearRedundantFeatures", '+',
                    "# of RBF-Like non linear redundant features", 0, 0, 1000);
    
    public FloatOption redundancyNoiseProbabilityOption
            = new FloatOption("redundancyNoiseProbability", 'w', "",
                    0.1, 0.0, 1.0);
    
    protected interface ClassFunction {
        
        public int determineClass(double attrib1, double attrib2);
    }
    
    protected static ClassFunction[] classificationFunctions = {
        // function 1
        new ClassFunction() {
            
            @Override
            public int determineClass(double attrib1, double attrib2) {
                return (attrib1 + attrib2 <= 5) ? 0 : 1;
            }
        },
        // function 2
        new ClassFunction() {
            
            @Override
            public int determineClass(double attrib1, double attrib2) {
                return (attrib1 + attrib2 <= 9) ? 0 : 1;
            }
        },
        // function 3
        new ClassFunction() {
            public int determineClass(double attrib1, double attrib2) {
                return (attrib1 + attrib2 <= 7) ? 0 : 1;
            }
        },
        // function 4
        new ClassFunction() {
            @Override
            public int determineClass(double attrib1, double attrib2) {
                return (attrib1 + attrib2 <= 9.5) ? 0 : 1;
            }
        }
    };
    
    protected InstancesHeader streamHeader;
    
    protected Random instanceRandom;
    
    protected boolean nextClassShouldBeZero;
    
    protected HashMap<Attribute, Attribute> redundantTo;
    
    protected HashMap<Attribute, Attribute> nonLinearCosineRedundantTo;
    
    protected HashMap<Attribute, Attribute> nonLinearRBFLikeRedundantTo;
    
    protected HashSet<Integer> indicesRelevants;
    
    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        this.indicesRelevants = new HashSet<>();
        this.redundantTo = new HashMap<>();
        this.nonLinearCosineRedundantTo = new FastHashMap<>();
        this.nonLinearRBFLikeRedundantTo = new FastHashMap<>();
        
        if (this.instanceRandom == null) {
            this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        }
        // generate header
        FastVector attributes = new FastVector();
        
        while (indicesRelevants.size() != 2) {
            indicesRelevants.add(Math.abs(this.instanceRandom.nextInt()) % (numRandomAttsOption.getValue() + 2));
        }
        
        for (int i = 0; i < numRandomAttsOption.getValue() + 2; i++) {
            attributes.add(new Attribute(("att" + (i))));
        }
        
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        
        int nextToBeCopied = 0;
        
        for (int i = 0; i < numRedundantFeaturesOption.getValue(); i++) {

            //picks a random feature to be used as source
            Attribute original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
            nextToBeCopied++;
            while (original.isNominal()) {
                original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
                nextToBeCopied++;
            }
            
            Attribute redundant = new Attribute("attrib" + attributes.size());
            attributes.addElement(redundant);
            
            redundantTo.put(redundant, original);
        }

        //make nonlinear redundant features here
        for (int i = 0; i < this.numCosineNonLinearRedundantFeaturesOption.getValue(); i++) {

            //picks a random feature to be used as source
            Attribute original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
            nextToBeCopied++;
            while (original.isNominal()) {
                original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
                nextToBeCopied++;
            }
            
            Attribute redundant = new Attribute("attrib" + attributes.size());
            attributes.addElement(redundant);
            this.nonLinearCosineRedundantTo.put(redundant, original);
            
        }
        
        for (int i = 0; i < this.numRBFlikeNonLinearRedundantFeaturesOption.getValue(); i++) {

            //picks a random feature to be used as source
            Attribute original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
            nextToBeCopied++;
            while (original.isNominal()) {
                original = (Attribute) attributes.get(nextToBeCopied % attributes.size());
                nextToBeCopied++;
            }
            
            Attribute redundant = new Attribute("attrib" + attributes.size());
            attributes.addElement(redundant);
            this.nonLinearRBFLikeRedundantTo.put(redundant, original);
            
        }
        
        FastVector classLabels = new FastVector();
        classLabels.addElement("groupA");
        classLabels.addElement("groupB");
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.streamHeader.setIndicesRelevants(getRelevantFeaturesInt());
        
        restart();
    }
    
    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }
    
    @Override
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }
    
    @Override
    public boolean hasMoreInstances() {
        return true;
    }
    
    @Override
    public boolean isRestartable() {
        return true;
    }
    
    @Override
    public InstanceExample nextInstance() {
        double attrib1 = 0, attrib2 = 0;
        int group = 0;
        boolean desiredClassFound = false;
        while (!desiredClassFound) {
            // generate attributes
            attrib1 = 10 * this.instanceRandom.nextDouble();
            attrib2 = 10 * this.instanceRandom.nextDouble();

            // determine class
            group = classificationFunctions[this.functionOption.getValue() - 1].
                    determineClass(attrib1, attrib2);
            if (!this.balanceClassesOption.isSet()) {
                desiredClassFound = true;
            } else // balance the classes
            {
                if ((this.nextClassShouldBeZero && (group == 0))
                        || (!this.nextClassShouldBeZero && (group == 1))) {
                    desiredClassFound = true;
                    this.nextClassShouldBeZero = !this.nextClassShouldBeZero;
                }
            }
        }
        //Add Noise
        if ((1 + (this.instanceRandom.nextInt(100)))
                <= this.noisePercentageOption.getValue()) {
            group = (group == 0 ? 1 : 0);
        }

        // construct instance
        InstancesHeader header = getHeader();
        Object indices[] = this.indicesRelevants.toArray();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);
        inst.setValue((int) indices[0], attrib1);
        inst.setValue((int) indices[1], attrib2);
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            if (!indicesRelevants.contains(i)) {
                inst.setValue(i, this.instanceRandom.nextDouble());
            }
        }
        
        for (Attribute att : this.redundantTo.keySet()) {
            double originalValue = inst.value(indexOfAttribute(inst, this.redundantTo.get(att)));
            if (this.instanceRandom.nextDouble() <= this.redundancyNoiseProbabilityOption.getValue()) {
                originalValue = 10 * this.instanceRandom.nextDouble();
            }
            inst.setValue(indexOfAttribute(inst, att), originalValue);
        }

        //buils nonlinear redundant features
        for (Attribute att : this.nonLinearCosineRedundantTo.keySet()) {
            double originalValue = inst.value(indexOfAttribute(inst, this.nonLinearCosineRedundantTo.get(att)));
            double max = 10.0;
            double min = 0.0;
            // COSINE
            //applies function
            double angle = 360 * originalValue / (max - min);
            double angleInRadians = angle * Math.PI / 180.0;
            originalValue = Math.cos(angleInRadians);
            
            if (this.instanceRandom.nextDouble() <= this.redundancyNoiseProbabilityOption.getValue()) {
                originalValue = (this.instanceRandom.nextBoolean() ? +1 : -1) * this.instanceRandom.nextDouble();
            }
            inst.setValue(indexOfAttribute(inst, att), originalValue);
        }
        
        for (Attribute att : this.nonLinearRBFLikeRedundantTo.keySet()) {
            double originalValue = inst.value(indexOfAttribute(inst, this.nonLinearRBFLikeRedundantTo.get(att)));
            double max = 10.0;
            double min = 0.0;
            // RBF-LIKE FUNCTION
            // A distribuição dos valores é uniforme, logo, 
            // o valor esperado é dado por
            // E(x) = (max(x) - min(x)) / 2
            double expectedValue = (max - min) / 2;
            originalValue = Math.pow(originalValue - expectedValue, 2.0);
            
            if (this.instanceRandom.nextDouble() <= this.redundancyNoiseProbabilityOption.getValue()) {
                originalValue = this.instanceRandom.nextDouble();
            }
            inst.setValue(indexOfAttribute(inst, att), originalValue);
        }
        
        inst.setClassValue(group);
        return new InstanceExample(inst);
    }
    
    @Override
    public void restart() {
        this.instanceRandom
                = new Random(this.instanceRandomSeedOption.getValue());
        this.nextClassShouldBeZero = false;

        //Outputs all relevant attributes' names
        System.out.print("relevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (this.indicesRelevants.contains(i)
                    && !this.redundantTo.keySet().contains(att)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        // irrelevant
        System.out.print("irrelevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (!this.indicesRelevants.contains(i)
                    && !this.redundantTo.keySet().contains(att)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        //redundant
        System.out.print("redundant = [");
        for (Attribute att : this.redundantTo.keySet()) {
            System.out.print(att.name() + ",");
        }
        System.out.print("] \t");

        //redundant to
        System.out.print("redundant to = [");
        for (Attribute att : this.redundantTo.keySet()) {
            System.out.print(att.name() + "<->" + redundantTo.get(att).name() + ",");
        }
        System.out.print("] \n");
        
    }
    
    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }
    
    private static int indexOfValue(String value, Object[] arr) {
        int index = Arrays.asList(arr).indexOf(value);
        return index;
    }
    
    private int indexOfAttribute(Instance instnc, Attribute att) {
        return instnc.indexOfAttribute(att);
    }



    /**
     * Method that results an array with the indices
     * of the ground-truth <b>relevant</b> features.
     *
     * @return an array with the indices of relevant features
     */
    public int[] getRelevantFeaturesInt() {
        int r[] = new int[indicesRelevants.size()];
        int index = 0;
        for(int i = 0; i < streamHeader.numAttributes() - 1; i++){
            if(indicesRelevants.contains(i)){
                r[index] = i;
                index++;
            }
        }
        return r;
    }

    /**
     * Method that results an array with the indices
     * of the ground-truth <b>irrelevant</b> features.
     *
     * @return an array with the indices of irrelevant features
     */
    public int[] getIrrelevantFeaturesInt() {
        int r[] = new int[streamHeader.numAttributes() - indicesRelevants.size() - 1];
        int index = 0;
        for(int i = 0; i < streamHeader.numAttributes() - 1; i++){
            if(!indicesRelevants.contains(i)){
                r[index] = i;
                index++;
            }
        }
        return r;
    }

}
