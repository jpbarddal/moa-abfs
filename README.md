# MOA-ABFS

## Boosting Decision Stumps for Dynamic Feature Selection on Data Streams (ABFS) - INFORMATION SYSTEMS SUBMISSION
This repository includes the code for the ABFS feature selector. The code here reflects the code used in the submission entitled **"Boosting Decision Stumps for Dynamic Feature Selection on Data Streams"** authored by *Jean Paul Barddal, Heitor Murilo Gomes, Fabrício Enembreck, Albert Bifet, and Bernhard Pfahringer*.
This paper has been submitted to Information Systems for refereeing.

In practice, this repository is a clone of the Massive Online Analysis (MOA) framework, with the addition of:

1. The code for the proposed method (`BoostingSelector.java`): this class represents the ABFS method proposed in the paper.
2. Abstract classes and interfaces for future feature selectors for data streams (`AbstractFeatureSelector.java` and `FeatureSelector.java`): these files contain the generic behavior that is shared between feature selectors. In practice, a feature selector should be able to be updated with labeled instances and return the currently selected subset of features at any time.
3. The gold-standard selector (`GoldStandardSelector.java`): Referred to as **ORACLE** in the paper, this selector selects the correct features for each instance if the generator has this information available.
3. The code for stability computation (`EvaluateFeatureSelectionStability.java`): This class is responsible for calculating the Stability of a feature selector on a data stream. An important parameter is the validation scheme to be adopted (bootstrap-, split- or cross-validation). Details about these schemes are available in the paper.
4. The code for Selection Accuracy computation (Part of the `AbstractFeatureSelector.java ` file): to facilitate the computation of feature selection-specific metrics, 
5. The scripts to reproduce the experiments discussed in the paper (`ABFSExperiments.java` and `StabilityExperiments.java`):
6. Synthetic data generators (`BG.java`, `BG2.java`, and `BG3.java`): These data generators have been introduced in one of our previous papers entitled ["A survey on feature drift adaptation: Definition, benchmark, challenges and future directions"](https://www.sciencedirect.com/science/article/pii/S0164121216301030?via%3Dihub) borrowing ideas from [Mark Hall's thesis](https://www.cs.waikato.ac.nz/~mhall/thesis.pdf) and were made available for MOA.

Upon paper acceptance, the authors plan to make the code available here part of the official repository of MOA (see details below).


## More about MOA
If you want to know more about MOA, please see the following paper:

> Albert Bifet, Geoff Holmes, Richard Kirkby, Bernhard Pfahringer (2010);
> MOA: Massive Online Analysis; Journal of Machine Learning Research 11: 1601-1604 


Also, the official GitHub repository for MOA can be found [here](https://github.com/waikato/moa).

![MOA][logo]

[logo]: http://moa.cms.waikato.ac.nz/wp-content/uploads/2014/11/LogoMOA.jpg "Logo MOA"
