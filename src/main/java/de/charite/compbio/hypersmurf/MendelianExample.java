/**
 * 
 */
package de.charite.compbio.hypersmurf;

import java.io.File;
import java.time.Duration;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.HyperSMURF;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.instance.SubsetByExpression;

/**
 * Mendelian Example
 * 
 * This class loads mendelian balanced and imbalanced datasets (arff file). And
 * then runs different configured hyperSMURF classifiers on it using a 10-fold
 * cytogenetic band-aware cross-validation. See {@link #main(String[])} how you
 * use the command-line to set the different datasets.
 *
 */
public class MendelianExample {

	/**
	 * We need a seed to make consistent predictions.
	 */
	private static int SEED = 42;

	/**
	 * The number of folds are predefined in the dataset
	 */
	private static int FOLDS = 10;

	/**
	 * Main method. Use {@code arg[0]} for data
	 * {@code Mendelian.balanced.arff.gz}, {@code arg[1]} for data
	 * {@code Mendelian.1_10.arff.gz}, {@code arg[2]} for data
	 * {@code Mendelian.1_100.arff.gz}, and {@code arg[3]} for data
	 * {@code Mendelian.1_1000.arff.gz}.
	 * 
	 * @param args
	 *            Four different paths to the different Mendelian datasets.
	 * @throws Exception
	 */
	public static void main(String[] args) throws Exception {

		// read the file from the first argument of the command line input
		ArffLoader reader = new ArffLoader();
		reader.setFile(new File(args[0]));
		Instances instances = reader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// setup the hyperSMURF classifier
		HyperSMURF clsHyperSMURF = new HyperSMURF();
		clsHyperSMURF.setNumIterations(2);
		clsHyperSMURF.setNumTrees(10);
		clsHyperSMURF.setDistributionSpread(0);
		clsHyperSMURF.setPercentage(0.0);
		clsHyperSMURF.setSeed(SEED);

		classify(clsHyperSMURF, instances);

		// Mendelian 1:10
		reader.setFile(new File(args[1]));
		instances = reader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// setup the hyperSMURF classifier
		clsHyperSMURF = new HyperSMURF();
		clsHyperSMURF.setNumIterations(5);
		clsHyperSMURF.setNumTrees(10);
		clsHyperSMURF.setDistributionSpread(0);
		clsHyperSMURF.setPercentage(100.0);
		clsHyperSMURF.setSeed(SEED);

		classify(clsHyperSMURF, instances);

		// Mendelian 1:100
		reader.setFile(new File(args[2]));
		instances = reader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// setup the hyperSMURF classifier
		clsHyperSMURF = new HyperSMURF();
		clsHyperSMURF.setNumIterations(10);
		clsHyperSMURF.setNumTrees(10);
		clsHyperSMURF.setDistributionSpread(3);
		clsHyperSMURF.setPercentage(200.0);
		clsHyperSMURF.setSeed(SEED);

		classify(clsHyperSMURF, instances);

		// Mendelian 1:1000
		reader.setFile(new File(args[3]));
		instances = reader.getDataSet();
		instances.setClassIndex(instances.numAttributes() - 1);

		// setup the hyperSMURF classifier
		clsHyperSMURF = new HyperSMURF();
		clsHyperSMURF.setNumIterations(10);
		clsHyperSMURF.setNumExecutionSlots(4);
		clsHyperSMURF.setNumTrees(10);
		clsHyperSMURF.setDistributionSpread(3);
		clsHyperSMURF.setPercentage(200.0);
		clsHyperSMURF.setSeed(SEED);

		// show the running time
		long startTime = System.nanoTime();

		classify(clsHyperSMURF, instances);

		long endTime = System.nanoTime();
		long duration = (endTime - startTime);

		System.out.println("Running time: " + Duration.ofNanos(duration).getSeconds() + " seconds!");

	}

	/**
	 * This methods uses an 10-fold CV (folds specified by the fold attribute)
	 * with the given classifier on the given instances and print out the
	 * performance to the console.
	 * 
	 * @param cls
	 *            The classifier for training.
	 * @param instances
	 *            Instances to train on.
	 * @throws Exception
	 */
	private static void classify(AbstractClassifier cls, Instances instances) throws Exception {
		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(instances);
		for (int n = 0; n < FOLDS; n++) {
			System.out.println("Training fold " + (n + 1) + " from " + FOLDS + "...");
			Instances train = getFold(instances, n, true);
			Instances test = getFold(instances, n, false);

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
		System.out.println("Folds: " + FOLDS);
		System.out.println("Seed: " + SEED);
		System.out.println();
		System.out.println(eval.toSummaryString("=== " + FOLDS + "-fold Cross-validation ===", false));
		System.out.println();
		System.out.println(eval.toClassDetailsString("=== Details ==="));

	}

	/**
	 * @param instances
	 *            The instances to use the method on. Must contain a attribute
	 *            named with fold which must be numerical.
	 * @param fold
	 *            The fold of interest (given by the fold attribute in the
	 *            instances)
	 * @param invert
	 *            if {@code true} the given fold will be selected. If
	 *            {@code false} All instances not of the given fold will be
	 *            selected
	 * @return
	 * @throws Exception
	 */
	private static Instances getFold(Instances instances, int fold, boolean invert) throws Exception {
		// filter on fold variable
		int indexFold = instances.attribute("fold").index();
		SubsetByExpression filterFold = new SubsetByExpression();
		if (invert)
			filterFold.setExpression("!(ATT" + (indexFold + 1) + " = " + fold + ")");
		else
			filterFold.setExpression("ATT" + (indexFold + 1) + " = " + fold);
		filterFold.setInputFormat(instances);
		Instances filtered = Filter.useFilter(instances, filterFold);

		// remove fold attribute
		filtered.deleteAttributeAt(indexFold);

		return filtered;
	}

}
