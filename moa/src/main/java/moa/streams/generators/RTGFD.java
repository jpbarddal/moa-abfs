package moa.streams.generators;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Random;

import com.yahoo.labs.samoa.instances.*;
import moa.core.FeatureSelectionUtils;
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
 * @author Jean Paul Barddal
 */
public class RTGFD extends AbstractOptionHandler implements
        InstanceStream {

    @Override
    public String getPurposeString() {
        return "Generates a stream based on a randomly generated tree.";
    }

    public IntOption treeRandomSeedOption = new IntOption("treeRandomSeed",
            'r', "Seed for random generation of tree.", 1);

    public IntOption instanceRandomSeedOption = new IntOption(
            "instanceRandomSeed", 'i',
            "Seed for random generation of instances.", 1);

    public IntOption numClassesOption = new IntOption("numClasses", 'c',
            "The number of classes to generate.", 2, 2, Integer.MAX_VALUE);

    public IntOption numNominalsOption = new IntOption("numNominals", 'o',
            "The number of nominal attributes to generate.", 5, 0,
            Integer.MAX_VALUE);

    public IntOption numNumericsOption = new IntOption("numNumerics", 'u',
            "The number of numeric attributes to generate.", 5, 0,
            Integer.MAX_VALUE);

    public IntOption numIrrelOption = new IntOption("numIrrel", '[',
            "The number of irrelevant nominal attributes to generate.", 5, 0,
            Integer.MAX_VALUE);

    public IntOption numValsPerNominalOption = new IntOption(
            "numValsPerNominal", 'v',
            "The number of values to generate per nominal attribute.", 5, 2,
            Integer.MAX_VALUE);

    public IntOption maxTreeDepthOption = new IntOption("maxTreeDepth", 'd',
            "The maximum depth of the tree concept.", 5, 0, Integer.MAX_VALUE);

    public IntOption firstLeafLevelOption = new IntOption(
            "firstLeafLevel",
            'l',
            "The first level of the tree above maxTreeDepth"
                    + " that can have leaves.",
            3, 0, Integer.MAX_VALUE);

    public FloatOption leafFractionOption = new FloatOption("leafFraction",
            'f',
            "The fraction of leaves per level from firstLeafLevel onwards.",
            0.15, 0.0, 1.0);

    protected static class Node implements Serializable {

        public int classLabel;
        public int splitAttIndex;
        public double splitAttValue;
        public Node[] children;
    }

    protected Node treeRoot;

    protected InstancesHeader streamHeader;

    protected Random instanceRandom;

    int[] relevantsInts;
    int[] irrelevantsInts;

    @Override
    public void prepareForUseImpl(TaskMonitor monitor,
                                  ObjectRepository repository) {
        monitor.setCurrentActivity("Preparing random tree...", -1.0);
        generateHeader();
        generateRandomTree();
        restart();
    }

    @Override
    public long estimatedRemainingInstances() {
        return -1;
    }

    @Override
    public boolean isRestartable() {
        return true;
    }

    @Override
    public void restart() {
        this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());

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
    public InstancesHeader getHeader() {
        return this.streamHeader;
    }

    @Override
    public boolean hasMoreInstances() {
        return true;
    }

    @Override
    public InstanceExample nextInstance() {
        double[] attVals = new double[this.numNominalsOption.getValue()
                + this.numNumericsOption.getValue()];
        InstancesHeader header = getHeader();
        Instance inst = new DenseInstance(header.numAttributes());
        inst.setDataset(header);
        for (int i = 0; i < attVals.length; i++) {
            Attribute att = this.streamHeader.attribute(i);
            attVals[i] = i < this.numNominalsOption.getValue()
                    ? this.instanceRandom.nextInt(this.numValsPerNominalOption.getValue())
                    : this.instanceRandom.nextDouble();
            inst.setValue(i, attVals[i]);
        }

        inst.setClassValue(classifyInstance(this.treeRoot, inst));

        for (int i = 0; i < irrelevantsInts.length; i++){
            int index = irrelevantsInts[i];
            Attribute att = this.streamHeader.attribute(index);
            if (att.isNominal()) {
                inst.setValue(index, Math.abs(this.instanceRandom.nextInt()) % att.numValues());
            } else {
                inst.setValue(index, this.instanceRandom.nextDouble());
            }
        }
        return new InstanceExample(inst);
    }

    public int classifyInstance(Instance inst) {
        return classifyInstance(this.treeRoot, inst);
    }

    protected int classifyInstance(Node node, Instance inst) {
        if (node.children == null) {
            return node.classLabel;
        }
        Attribute attSplit = this.streamHeader.attribute(node.splitAttIndex);
        if (attSplit.isNominal()) {
            int value = (int) inst.value(indexOfAttribute(inst, attSplit));
            return classifyInstance(
                    node.children[value], inst);
        }
        double value = inst.value(indexOfAttribute(inst, attSplit));
        return classifyInstance(
                node.children[value < node.splitAttValue ? 0 : 1], inst);
    }

    protected void generateHeader() throws IllegalArgumentException {
        if (this.instanceRandom == null) {
            this.instanceRandom = new Random(this.instanceRandomSeedOption.getValue());
        }

        HashSet<Integer> indicesIrrelevants = new HashSet<>();
        if (numIrrelOption.getValue() < numNominalsOption.getValue() + numNumericsOption.getValue()) {
            while (indicesIrrelevants.size() != numIrrelOption.getValue()) {
                int value = this.instanceRandom.nextInt(this.numNominalsOption.getValue() + this.numNumericsOption.getValue());
                if (!indicesIrrelevants.contains(value)) {
                    indicesIrrelevants.add(value);
                }
            }
        } else {
            throw new IllegalArgumentException("numnominals + numnumerics should be greater than numirrels!");
        }

        int numFeatures = this.numNumericsOption.getValue() + this.numNominalsOption.getValue();
        relevantsInts = new int[numFeatures - indicesIrrelevants.size()];
        irrelevantsInts = new int[indicesIrrelevants.size()];
        int iRelevant = 0;
        int iIrrelevant = 0;
        for(int i = 0; i < numFeatures; i++){
            if(indicesIrrelevants.contains(i)){
                irrelevantsInts[iIrrelevant] = i;
                iIrrelevant++;
            }else{
                relevantsInts[iRelevant] = i;
                iRelevant++;
            }
        }

        FastVector attributes = new FastVector();
        FastVector nominalAttVals = new FastVector();
        for (int i = 0; i < this.numValsPerNominalOption.getValue(); i++) {
            nominalAttVals.addElement("value" + (i + 1));
        }

        for (int i = 0; i < this.numNominalsOption.getValue(); i++) {
            Attribute att = new Attribute("attrib" + i, nominalAttVals);
            attributes.addElement(att);
        }
        for (int i = 0; i < this.numNumericsOption.getValue(); i++) {
            Attribute att = new Attribute("attrib" + attributes.size());
            attributes.addElement(att);
        }

        FastVector classLabels = new FastVector();
        for (int i = 0; i < this.numClassesOption.getValue(); i++) {
            classLabels.addElement("class" + (i + 1));
        }
        attributes.addElement(new Attribute("class", classLabels));
        this.streamHeader = new InstancesHeader(new Instances(
                getCLICreationString(InstanceStream.class), attributes, 0));
        this.streamHeader.setClassIndex(this.streamHeader.numAttributes() - 1);
        this.streamHeader.setIndicesRelevants(relevantsInts);
    }

    protected void generateRandomTree() {
        Random treeRand = new Random(this.treeRandomSeedOption.getValue());
        ArrayList<Integer> nominalAttCandidates = new ArrayList<>(
                this.numNominalsOption.getValue());
        for (int i = 0; i < this.numNominalsOption.getValue(); i++) {
            nominalAttCandidates.add(i);
        }
        double[] minNumericVals = new double[this.numNumericsOption.getValue()];
        double[] maxNumericVals = new double[this.numNumericsOption.getValue()];
        for (int i = 0; i < this.numNumericsOption.getValue(); i++) {
            minNumericVals[i] = 0.0;
            maxNumericVals[i] = 1.0;
        }
        this.treeRoot = generateRandomTreeNode(0, nominalAttCandidates,
                treeRand);
    }

    protected Node generateRandomTreeNode(int currentDepth,
                                          ArrayList<Integer> nominalAttCandidates,
                                          Random treeRand) {
        if ((currentDepth >= this.maxTreeDepthOption.getValue())
                || ((currentDepth >= this.firstLeafLevelOption.getValue())
                && (this.leafFractionOption.getValue()
                >= (1.0 - treeRand.nextDouble())))) {
            Node leaf = new Node();
            leaf.classLabel = treeRand.nextInt(this.numClassesOption.getValue());
            return leaf;
        }
        Node node = new Node();
        int randomIndex = Math.abs(treeRand.nextInt()) % (relevantsInts.length);
        int indexAtt = relevantsInts[randomIndex];
        Attribute att = this.streamHeader.attribute(indexAtt);
        if (att.isNominal()) {
            node.splitAttIndex = indexAtt;
            node.children = new Node[this.numValsPerNominalOption.getValue()];
            ArrayList<Integer> newNominalCandidates = new ArrayList<>(
                    nominalAttCandidates);
            newNominalCandidates.remove(new Integer(node.splitAttIndex));

            newNominalCandidates.trimToSize();
            for (int i = 0; i < node.children.length; i++) {
                node.children[i] = generateRandomTreeNode(currentDepth + 1,
                        newNominalCandidates,
                        treeRand);
            }
        } else {
            node.splitAttIndex = indexAtt;
            double minVal = 0.0;
            double maxVal = 1.0;
            node.splitAttValue = ((maxVal - minVal) * treeRand.nextDouble())
                    + minVal;
            node.children = new Node[2];
            node.children[0] = generateRandomTreeNode(currentDepth + 1,
                    nominalAttCandidates,
                    treeRand);
            node.children[1] = generateRandomTreeNode(currentDepth + 1,
                    nominalAttCandidates,
                    treeRand);
        }
        return node;
    }

    @Override
    public void getDescription(StringBuilder sb, int indent) {
    }

    private static int indexOfValue(String value, Object[] arr) {
        int index = Arrays.asList(arr).indexOf(value);
        return index;
    }

    private int indexOfAttribute(Instance instnc, Attribute att) {
        for (int i = 0; i < instnc.numAttributes(); i++) {
            Attribute chk = instnc.attribute(i);
            if (instnc.attribute(i).name().equals(att.name())) {
                return i;
            }
        }
        return -1;
    }
}
