/**
 * 
 */
package de.charite.compbio.hypersmurf;

import java.io.File;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.HyperSMURF;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ArffLoader;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.instance.SubsetByExpression;

public class MendelianExample {

	/**
	 * We need a seed to make consistent predictions.
	 */
	private static int SEED = 42;

	/**
	 * The number of folds are predifined in the dataset
	 */
	private static int FOLDS = 10;

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

	}

	private static void classify(AbstractClassifier cls, Instances instances) throws Exception {
		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(instances);
		for (int n = 0; n < FOLDS; n++) {
			System.out.println("Training fold " + (n+1) + " from " + FOLDS + "...");
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
	 * @param fold
	 * @param invert
	 * @return
	 * @throws Exception
	 */
	private static Instances getFold(Instances instances, int fold, boolean invert) throws Exception {
		// filter on fold variable
		SubsetByExpression filterFold = new SubsetByExpression();
		int indexFold = instances.attribute("fold").index();
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
