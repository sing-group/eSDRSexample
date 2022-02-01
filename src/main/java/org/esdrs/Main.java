/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package org.esdrs;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import org.bdp4j.types.Dataset;
import org.nlpa.transformers.dataset.DatasetFeatureRepresentation;
import org.nlpa.transformers.dataset.eSDRS;
import weka.core.converters.ArffLoader;

/**
 *
 * @author María Novo
 */
public class Main {

    /**
     * Main method
     *
     * @param args Args needed to main execution
     */
    public static void main(String[] args) {
        // Theorethical evaluation
        String file = "exampleFile.arff";
        try {
            realScenario(file, 3, 0.85, 25, 75);
        } catch (IOException ex) {
            System.out.println("File " + file + " does not exist.");
        }

    }

    /**
     * Generate generalizated files in a real scenario
     *
     * @param fileName The name of file to generalize
     * @param maxDegree Maximum distance between synsets in Babelnet
     * @param requiredSimilarity Ratio of similarity needed to consider that two
     * synsets belongs to the same class
     * @param ntest Percentage of instances in test dataset
     * @param ntrain Percentage of instances in training dataset
     * @throws IOException Exception if file does not exists
     */
    public static void realScenario(String fileName, int maxDegree, double requiredSimilarity, int ntest, int ntrain) throws IOException {

        BufferedReader reader;
        try {
            /* Load file to generalize */
            reader = new BufferedReader(new FileReader(fileName));
            ArffLoader.ArffReader arff = new ArffLoader.ArffReader(reader);;
            Dataset originalDataset = new Dataset(arff.getData());

            /* Splitting original dataset in train and test files*/
            Dataset[] stratifiedDataset = originalDataset.split(true, ntest, ntrain);
            Dataset testingDataset = stratifiedDataset[0];
            Dataset trainingDataset = stratifiedDataset[1];

            /* Generalize training dataset */
            trainingDataset.filterColumns("id|^bn|target");
            eSDRS gESDRS = new eSDRS(maxDegree, Dataset.COMBINE_SUM, requiredSimilarity);
            gESDRS.setDatatype(eSDRS.Datatype.APPEARENCES_NUMBER);
            System.out.println("Max Distance: " + gESDRS.getMaxDegree() + " - Match rate: " + gESDRS.getRequiredSimilarity());
            Dataset generalizatedTrainDataset = gESDRS.transform(trainingDataset);

            /* Convert generalizatedTrainDataset in to a csv file*/
            int degree = gESDRS.getMaxDegree();
            int rs = (int) (gESDRS.getRequiredSimilarity() * 100);
            generalizatedTrainDataset.filterColumns("^bn|id|target");
            generalizatedTrainDataset.setOutputFile("MD" + degree + "_RS0" + rs + "_Train" + ntrain + "_Reduced.csv");
            generalizatedTrainDataset.generateCSV();

            /* Generalize testing dataset from generalized training dataset */
            System.out.println("Start DatasetFeatureRepresentation");
            DatasetFeatureRepresentation dfr = new DatasetFeatureRepresentation(generalizatedTrainDataset);
            Dataset generalizatedTestDataset = dfr.transform(testingDataset);

            /* Convert generalizatedTestDataset in to a csv file*/
            generalizatedTestDataset.setOutputFile("MD" + degree + "_RS0" + rs + "_Test" + ntest + "_Reduced.csv");
            generalizatedTestDataset.generateCSV();
            System.out.println("End DatasetFeatureRepresentation");
        } catch (FileNotFoundException ex) {
            System.out.println("The file " + fileName + "does not exist");
        }
    }

}
