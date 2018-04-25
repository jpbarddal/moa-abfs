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
public class BG2 extends AbstractOptionHandler implements
        InstanceStream {

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public FlagOption balanceClassesOption = new FlagOption("balanceClasses",
            'b', "Balance the number of instances of each class.");

    public IntOption noisePercentageOption = new IntOption("noisePercentage",
            'n', "Percentage of noise to add to the data.", 10, 0, 100);

    public StringOption relevantFeaturesOption
            = new StringOption("relevantFeatures", 'f', "", "");

    public IntOption numFeaturesOption
            = new IntOption("numFeatures", 'F', "", 0, 0, 1024);

    public IntOption numRedundantFeaturesOption
            = new IntOption("numRedundantFeatures", 'R', "", 0, 0, 1000);

    public FloatOption redundancyNoiseProbabilityOption
            = new FloatOption("redundancyNoiseProbability", 'w', "",
                    0.1, 0.0, 1.0);

    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    protected boolean nextClassShouldBeFalse;

    FastVector classLabels;
    FastVector values;

    int[] relevantsInts;
    int[] irrelevantsInts;

    @Override
    protected void prepareForUseImpl(TaskMonitor monitor,
            ObjectRepository repository) {

        // generate header
        FastVector attributes = new FastVector();
        this.values = new FastVector();
        values.add("F");
        values.add("T");
        

        HashSet<Integer> indicesRelevants = new HashSet<>();
        String indices[] = this.relevantFeaturesOption.getValue().split(";");
        for (String strIndex : indices) {
            if (!strIndex.equals("")) {
                int index = Integer.parseInt(strIndex);
                indicesRelevants.add(index);
            }
        }

        if(indicesRelevants.size() != 3) throw new IllegalArgumentException("This generator requires " +
                "exactly 3 relevant features.");

        relevantsInts = new int[indicesRelevants.size()];
        int ii = 0;
        for(Integer index : indicesRelevants){
            relevantsInts[ii] = index;
            ii++;
        }

        irrelevantsInts = new int[numFeaturesOption.getValue() - relevantsInts.length];
        int index = 0;
        for (int i = 0; i < numFeaturesOption.getValue(); i++) {
            if(!FeatureSelectionUtils.contains(i, relevantsInts)){
                irrelevantsInts[index] = i;
                index++;
            }
            attributes.add(new Attribute(("attrib" + (i)), values));
        }

        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());

        this.classLabels = new FastVector();
        classLabels.addElement("false");
        classLabels.addElement("true");
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
        if ((1 + (Math.abs(this.instanceRandom.nextInt(100))))
                <= this.noisePercentageOption.getValue()) {
            classValue = (classValue == false ? true : false);
        }

        // construct instance
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);
        for (int i = 0; i < atts.length; i++) {
            String value = atts[i] ? "T" : "F";
            inst.setValue(i, value.equals("T") ? 1 : 0);
        }

        inst.setClassValue(classValue ? 1 : 0);
        return new InstanceExample(inst);

    }

    @Override
    public void restart() {
        this.instanceRandom
                = new Random(this.instanceRandomSeedOption.getValue());
//        System.out.println("->" + instanceRandom.nextDouble());
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
        boolean atts[] = new boolean[numFeaturesOption.getValue()];
        for (int i = 0; i < atts.length; i++) {
            atts[i] = this.instanceRandom.nextBoolean();
        }
        int indexA = relevantsInts[0];
        int indexB = relevantsInts[1];
        int indexC = relevantsInts[2];
        int random = Math.abs(this.instanceRandom.nextInt()) % 3;
        if (classValue == true) {
            if (!(atts[indexA] && atts[indexB]
                    || atts[indexA] && atts[indexC]
                    || atts[indexB] && atts[indexC])) {
                //um dos E' não está sendo atendido
                //sorteia um dos E's
                //e força que ele seja verdadeiro                
                switch (random) {
                    case 0:
                        atts[indexA] = true;
                        atts[indexB] = true;
                        break;
                    case 1:
                        atts[indexA] = true;
                        atts[indexC] = true;
                        break;
                    case 2:
                    default:
                        atts[indexB] = true;
                        atts[indexC] = true;
                        break;
                }
            }
        } else {
            //condições que devem ser satisfeitas:
            //~alpha V ~beta
            //~alpha V ~epsilon
            //~beta V ~epsilon

            //na prática, ocorre que apenas uma das três features relevantes
            // pode ser verdadeira, logo, sorteia-se uma e 
            //as demais serão setadas para falso
//            int random = this.instanceRandom.nextInt() % 3;
            switch (random) {
                case 0:
                    atts[indexA] = true;
                    atts[indexB] = false;
                    atts[indexC] = false;
                    break;
                case 1:
                    atts[indexB] = true;
                    atts[indexA] = false;
                    atts[indexC] = false;
                    break;
                case 2:
                default:
                    atts[indexC] = true;
                    atts[indexA] = false;
                    atts[indexB] = false;
                    break;
            }

        }
//        System.out.println(Arrays.toString(atts));
        return atts;

    }

}
