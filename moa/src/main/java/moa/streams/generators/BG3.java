package moa.streams.generators;

import com.github.javacliparser.FlagOption;
import com.github.javacliparser.FloatOption;
import com.github.javacliparser.IntOption;
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
public class BG3 extends AbstractOptionHandler implements
        InstanceStream {

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");

    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 10, 0, 100);

    public IntOption numIrrelevantFeaturesOption
            = new IntOption("numIrrelevantFeatures", 'F', "", 0, 0, 1024);

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
            String value = atts[i] ? "T" : "F";
            inst.setValue(i, indexOfValue(value, values.toArray()));
        }

        inst.setClassValue(classValue ? indexOfValue("groupA", classLabels.toArray()) :
                indexOfValue("groupB", classLabels.toArray()));
        return new InstanceExample(inst);
    }

    @Override
    public void restart() {
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        this.nextClassShouldBeFalse = false;

        // generate header
        values = new FastVector();
        values.add("T");
        values.add("F");
        FastVector attributes = new FastVector();

        TreeSet<Integer> indicesRelevants = new TreeSet<>();

        int numFeatures = 3 + numIrrelevantFeaturesOption.getValue() + 1;
        int classIndex = numFeatures - 1;

        while (indicesRelevants.size() != 3) {
            int ii = Math.abs(this.instanceRandom.nextInt())
                    % (numIrrelevantFeaturesOption.getValue() + 3);
            if(ii != classIndex) indicesRelevants.add(ii);
        }

        relevantsInts = new int[indicesRelevants.size()];
        int ii = 0;
        for(Integer index : indicesRelevants){
            relevantsInts[ii] = index;
            ii++;
        }

        irrelevantsInts = new int[numFeatures - relevantsInts.length - 1];
        int index = 0;
        for (int i = 0; i < numFeatures; i++) {
            if(!FeatureSelectionUtils.contains(i, relevantsInts) && i != classIndex){
                irrelevantsInts[index] = i;
                index++;
            }
            attributes.add(new Attribute(("attrib" + (i)), values));
        }

        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());

        classLabels = new FastVector();
        classLabels.addElement("groupA");
        classLabels.addElement("groupB");
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));

        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.streamHeader.setIndicesRelevants(relevantsInts);

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
        return "Generates a concept accordingly to binary variables.";
    }

    private boolean[] createInstance(boolean classValue) {
        boolean atts[] = new boolean[numIrrelevantFeaturesOption.getValue() + 3];
        for (int i = 0; i < atts.length; i++) {
            atts[i] = this.instanceRandom.nextBoolean();
        }
        int indexA = relevantsInts[0];
        int indexB = relevantsInts[1];
        int indexC = relevantsInts[2];
        int random = Math.abs(this.instanceRandom.nextInt()) % 3;
        if (classValue == true) {
            //todas as variáveis devem ser TRUE ou FALSE            
            switch (random) {
                case 0:
//                    tudo true
                    atts[indexA] = atts[indexB] = atts[indexC] = true;
                    break;
                case 1:
                default:
                    //tudo false
                    atts[indexA] = atts[indexB] = atts[indexC] = false;
                    break;

            }
        } else {
//            pode acontecer qualquer coisa, exceto as condições acima
            boolean test = (atts[indexA] && atts[indexB] && atts[indexC])
                    || (!atts[indexA] && !atts[indexB] && !atts[indexC]);
            while (test) {
                int index = (random == 0 ? indexA : (random == 1 ? indexB : indexC));
                atts[index] = !atts[index];
                test = (atts[indexA] && atts[indexB] && atts[indexC])
                        || (!atts[indexA] && !atts[indexB] && !atts[indexC]);
            }
        }
        return atts;

    }

    private int indexOfValue(String value, Object[] arr) {
        int index = Arrays.asList(arr).indexOf(value);
        return index;
    }

}
