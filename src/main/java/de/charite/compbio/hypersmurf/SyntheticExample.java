package de.charite.compbio.hypersmurf;

import java.util.Arrays;
import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.HyperSMURF;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.datagenerators.classifiers.classification.RDG1;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

/**
 * Synthetic Example
 * 
 * This class generates a sythetic imbalanced dataset using the {@link RDG1} data generator and an own prcedure to
 * imbalance the data. Then we can train {@link HyperSMURF} or any other classifier on the data (like
 * {@link RandomForest}).
 *
 */
public class SyntheticExample {

	/**
	 * We need a seed to make consistent predictions.
	 */
	private static int SEED = 42;

	/**
	 * Main function. Generates an imbalanced dataset and shows the performance of hyperSMURF and a RandomForest
	 * classifier using a 5-fold cross-validation.
	 * 
	 * @param args
	 *            Not needed
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// Generate synthetic data
		Instances imbalancedInstances = generateSyntheticData(10000, 50);

		// setup the hyperSMURF classifier
		HyperSMURF clsHyperSMURF = new HyperSMURF();
		clsHyperSMURF.setNumIterations(10);
		clsHyperSMURF.setNumTrees(10);
		clsHyperSMURF.setDistributionSpread(0);
		clsHyperSMURF.setPercentage(200.0);
		clsHyperSMURF.setSeed(SEED);

		// classify hyperSMURF
		classify(clsHyperSMURF, imbalancedInstances, 5);

		// setup a RF classifier
		RandomForest clsRF = new RandomForest();
		clsRF.setNumIterations(10);
		clsRF.setSeed(SEED);

		// classify RF
		classify(clsRF, imbalancedInstances, 5);

	}

	/**
	 * This methods uses an k-fold CV with the given classifier on the given instances and print out the performance in
	 * the console.
	 * 
	 * @param cls
	 *            The classifier for training.
	 * @param instances
	 *            Instances to train on.
	 * @param folds
	 *            Number of folds should be used for CV.
	 * @throws Exception
	 */
	private static void classify(AbstractClassifier cls, Instances instances, int folds) throws Exception {
		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(instances);
		for (int n = 0; n < folds; n++) {
			System.out.println("Training fold " + (n+1) + " from " + folds + "...");
			Instances train = instances.trainCV(folds, n);
			Instances test = instances.testCV(folds, n);

			// build and evaluate classifier
			Classifier clsCopy = AbstractClassifier.makeCopy(cls);
			clsCopy.buildClassifier(train);
			eval.evaluateModel(clsCopy, test);

			// add predictions
			AddClassification filter = new AddClassification();
			filter.setClassifier(cls);
			filter.setOutputClassification(true);
			filter.setOutputDistribution(true);
			filter.setOutputErrorFlag(true);
			filter.setInputFormat(train);
			Filter.useFilter(train, filter); // trains the classifier
			Instances pred = Filter.useFilter(test, filter); // perform
																// predictions
																// on test set
			if (predictedData == null)
				predictedData = new Instances(pred, 0);
			for (int j = 0; j < pred.numInstances(); j++)
				predictedData.add(pred.instance(j));
		}

		// output evaluation
		System.out.println();
		System.out.println("=== Setup ===");
		System.out.println("Classifier: " + cls.getClass().getName() + " " + Utils.joinOptions(cls.getOptions()));
		System.out.println("Dataset: " + instances.relationName());
		System.out.println("Folds: " + folds);
		System.out.println("Seed: " + SEED);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + folds + "-fold Cross-validation ===", false));
		System.out.println();
		System.out.println(eval.toClassDetailsString("=== Details ==="));

	}

	/**
	 * This method generates an imbalanced dataset using the {@link RDG1} data generator and additional procedures to
	 * imbalance it.
	 * 
	 * @param numExamples
	 *            Number of (theoretical) instances {@link RDG1} should generate. Will not be the final number of
	 *            instances.
	 * @param numMinority
	 *            The total number of all instances of the first class.
	 * @return An imbalanced dataset with numMinority numbers of the first class.
	 * @throws Exception
	 */
	private static Instances generateSyntheticData(int numExamples, int numMinority) throws Exception {

		System.out.println("Start generating synthetic data...");
		// Generate a synthetic Example using the Weka
		RDG1 dataGenerator = new RDG1();
		dataGenerator.setRelationName("SyntheticData");
		dataGenerator.setNumExamples(numExamples);
		dataGenerator.setNumAttributes(20);
		dataGenerator.setNumNumeric(20);
		dataGenerator.setSeed(SEED);
		dataGenerator.defineDataFormat();
		Instances instances = dataGenerator.generateExamples();

		// set the index to last attribute
		instances.setClassIndex(instances.numAttributes() - 1);

		// randomize the data
		Random random = new Random(SEED);
		instances.randomize(random);

		// data before imbalancing
		int[] counts = countClasses(instances);
		System.out.println("Before imbalancing: " + Arrays.toString(counts));

		// imbalance data
		int numberOfClassOne = numMinority;
		Instances imbalancedInstances = new Instances(instances, counts[1] + numberOfClassOne);
		for (int i = 0; i < instances.numInstances(); i++) {
			if (instances.get(i).classValue() == 0.0) {
				if (numberOfClassOne != 0) {
					imbalancedInstances.add(instances.get(i));
					numberOfClassOne--;
				}
			} else {
				imbalancedInstances.add(instances.get(i));
			}
		}
		imbalancedInstances.randomize(random);
		counts = countClasses(imbalancedInstances);
		System.out.println("After imbalancing: " + Arrays.toString(counts));

		return imbalancedInstances;
	}

	/**
	 * Helper that counts the class sizes.
	 * 
	 * @param instances
	 * @return The class sizes.
	 */
	private static int[] countClasses(Instances instances) {
		int[] counts = new int[instances.numClasses()];
		for (Instance instance : instances) {
			if (instance.classIsMissing() == false) {
				counts[(int) instance.classValue()]++;
			}
		}
		return counts;
	}

}
