package moa.streams.generators;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
import com.github.javacliparser.StringOption;
import com.yahoo.labs.samoa.instances.*;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;
import java.util.TreeSet;

import moa.core.FeatureSelectionUtils;
import moa.core.InstanceExample;
import moa.core.ObjectRepository;
import moa.options.AbstractOptionHandler;
import moa.streams.InstanceStream;
import moa.tasks.TaskMonitor;
import org.apache.commons.collections15.map.FastHashMap;
import weka.core.FastVector;

/**
 *
 * @author Jean Paul Barddal
 */
public class BG extends AbstractOptionHandler implements
        InstanceStream {

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");

    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 10, 0, 100);

    public IntOption numFeaturesOption
            = new IntOption("numFeatures", 'F', "", 0, 0, 1024);

    public StringOption relevantFeaturesOption
            = new StringOption("relevantFeatures", 'f', "", "");

    public IntOption numRedundantFeaturesOption
            = new IntOption("numRedundantFeatures", 'R', "", 0, 0, 1000);

    public FloatOption redundancyNoiseProbabilityOption
            = new FloatOption("redundancyNoiseProbability", 'w', "",
                    0.1, 0.0, 1.0);

    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    protected boolean nextClassShouldBeFalse;


    FastVector values;
    FastVector classLabels;

    int[] relevantsInts;
    int[] irrelevantsInts;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {
        // generate header
        FastVector attributes = new FastVector();
        this.values = new FastVector();
        values.add("T");
        values.add("F");

        HashSet<Integer> indicesRelevants = new HashSet<>();
//        namesRelevants = new HashSet<>();
        String indices[] = this.relevantFeaturesOption.getValue().split(";");

        for (String strIndex : indices) {
            if (!strIndex.equals("")) {
                int index = Integer.parseInt(strIndex);
                indicesRelevants.add(index);
            }
        }
        relevantsInts = new int[indicesRelevants.size()];
        int ii = 0;
        for(Integer index : indicesRelevants){
            relevantsInts[ii] = index;
            ii++;
        }

        ii = 0;
        irrelevantsInts = new int[numFeaturesOption.getValue() - indicesRelevants.size()];
        for (int i = 0; i < numFeaturesOption.getValue(); i++) {
            attributes.add(new Attribute(("attrib" + (i)), values));
            if (!indicesRelevants.contains(i)) {
                irrelevantsInts[ii] = i;
                ii++;
            }
        }

        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());

        this.classLabels = new FastVector();
        classLabels.addElement("groupA");
        classLabels.addElement("groupB");
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.streamHeader.setIndicesRelevants(relevantsInts);

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
        boolean classValue = nextClassShouldBeFalse == false ? true : false;

        if (balanceClassesOption.isSet()) {
            nextClassShouldBeFalse = !nextClassShouldBeFalse;
        } else {
            nextClassShouldBeFalse = this.instanceRandom.nextBoolean();
        }

        // generate attributes
        boolean atts[] = createInstance(classValue);

        //Add Noise
        if ((1 + (this.instanceRandom.nextInt(100)))
                <= this.noisePercentageOption.getValue()) {
            classValue = (classValue == false ? true : false);
        }

        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);
        for (int i = 0; i < atts.length; i++) {
//            int value = atts[i] ? 1 : 0;
            String value = atts[i] ? "T" : "F";
            inst.setValue(i, indexOfValue(value, values.toArray()));
        }

        inst.setClassValue(classValue ? indexOfValue("groupA", classLabels.toArray()) : indexOfValue("groupB", classLabels.toArray()));
//        System.out.println(inst);
        return new InstanceExample(inst);
//        return (Example<Instance>) inst;
    }

    @Override
    public void restart() {
        this.instanceRandom
                = new Random(this.instanceRandomSeedOption.getValue());
        this.nextClassShouldBeFalse = false;


        //Outputs all relevant attributes' names
        System.out.print("relevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (FeatureSelectionUtils.contains(i, relevantsInts)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \t");

        // irrelevant
        System.out.print("irrelevant = [");
        for (int i = 0; i < streamHeader.numAttributes() - 1; i++) {
            Attribute att = streamHeader.attribute(i);
            if (FeatureSelectionUtils.contains(i, irrelevantsInts)) {
                System.out.print(att.name() + ",");
            }
        }
        System.out.print("] \n");

    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    public String getPurposeString() {
        return "Generates a AND concept accordingly to binary variables.";
    }

    private boolean[] createInstance(boolean classValue) {
        boolean atts[] = new boolean[numFeaturesOption.getValue()];
        for (int i = 0; i < atts.length; i++) {
            atts[i] = this.instanceRandom.nextBoolean();
        }
        if (classValue == true) {
            String indices[] = this.relevantFeaturesOption.getValue().split(";");
            for (String strIndex : indices) {
                if (!strIndex.equals("")) {
                    int index = Integer.parseInt(strIndex);
                    atts[index] = true;
                }
            }
        } else {
//            //at least one of the attributes must be set to false
//            //therefore, we randomize which attributes will be set to false
            String indices[] = this.relevantFeaturesOption.getValue().split(";");
            boolean obtainedClassValue = false;
            while (!obtainedClassValue) {
                for (String strIndex : indices) {
                    if (!strIndex.equals("")) {
                        int index = Integer.parseInt(strIndex);
                        if (this.instanceRandom.nextDouble() < 0.5) {
                            atts[index] = false;
                        }
                    }
                }

                for (String strIndex : indices) {
                    if (!strIndex.equals("")) {
                        int index = Integer.parseInt(strIndex);
                        if (!atts[index]) {
                            obtainedClassValue = true;
                        }
                    }
                }
            }

        }

        return atts;
    }

    private static int indexOfValue(String value, Object[] arr) {
        int index = Arrays.asList(arr).indexOf(value);
        return index;
    }

    private int indexOfAttribute(Instance instnc, Attribute att) {
        for (int i = 0; i < instnc.numAttributes(); i++) {
            if (instnc.attribute(i).name().equals(att.name())) {
                return i;
            }
        }
        return -1;
    }

}
