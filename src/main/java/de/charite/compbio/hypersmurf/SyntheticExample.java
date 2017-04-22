package de.charite.compbio.hypersmurf;

import java.util.Random;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.HyperSMURF;
import weka.core.Instances;
import weka.core.Utils;
import weka.datagenerators.classifiers.classification.Agrawal;
import weka.filters.Filter;
import weka.filters.supervised.attribute.AddClassification;

/**
 * SyntheticExample
 *
 */
public class SyntheticExample {

	private static int SEED = 42;
	private static int FOLDS = 5;

	public static void main(String[] args) throws Exception {

		System.out.println("Start generating synthetic data...");
		// Generate a synthetic Example using the Weka
		Agrawal dataGenerator = new Agrawal();
		dataGenerator.setRelationName("SyntheticData");
		dataGenerator.setNumExamples(100000);
		dataGenerator.setSeed(SEED);
		dataGenerator.defineDataFormat();
		Instances instances = dataGenerator.generateExamples();
		// set the index to group
		instances.setClassIndex(instances.numAttributes() - 1);
		// randomize the data
		Random random = new Random(SEED);

		instances.randomize(random);

		// setup the hyperSMURF classifier
		HyperSMURF cls = new HyperSMURF();
		cls.setNumIterations(10);
		cls.setPercentage(100.0);

		// perform cross-validation and add predictions
		Instances predictedData = null;
		Evaluation eval = new Evaluation(instances);
		for (int n = 0; n < FOLDS; n++) {
			System.out.println("Training fold " + n + " from " + FOLDS + "...");
			Instances train = instances.trainCV(FOLDS, n);
			Instances test = instances.testCV(FOLDS, n);

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

	}

}
