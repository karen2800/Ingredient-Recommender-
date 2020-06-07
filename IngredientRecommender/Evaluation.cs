using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace IngredientRecommender
{
    struct Results
    {
        public double TP; // True Positive
        public double TN; // True Negative
        public double FP; // False Positive
        public double FN; // False Negative

        public Results(int n)
        {
            TP = 0.0;
            TN = 0.0;
            FP = 0.0;
            FN = 0.0;
        }

        public double getAccuracy()
        {
            return (TP + TN) / (TP + TN + FP + FN);
        }
        public double getBalancedAccuracy()
        {
            return getSensitivity() / getSpecificity();
        }
        public double getSensitivity()
        {
            return TP / (TP + FN);
        }
        public double getSpecificity()
        {
            return TN / (TN + FP);
        }
        public double getPrecision()
        {
            return TP / (TP + FP);
        }
        public double getRecall()
        {
            return TP / (TP + FN);
        }
        public double getF1()
        {
            return 2 * (getPrecision() * getRecall()) / (getPrecision() + getRecall());
        }
        public double getMCC()
        {
            return (TP * TN - FP * FN) / (Math.Sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)));
        }
        public void ShowResults()
        {
            Console.WriteLine("\nBalanced accuracy:\t" + getBalancedAccuracy());
            Console.WriteLine("Sensitivity:\t\t" + getSensitivity());
            Console.WriteLine("Specificity:\t\t" + getSpecificity());

            Console.WriteLine("\nAccuracy:\t\t" + getAccuracy());
            Console.WriteLine("Precision:\t\t" + getPrecision());
            Console.WriteLine("Recall:\t\t\t" + getRecall());
            Console.WriteLine("\nF1:\t\t\t" + getF1());
            Console.WriteLine("MCC:\t\t\t" + getMCC());
            Console.WriteLine();
        }
    }
    class Evaluation
    {
        // Evaluate Modified KNN (dynamic k)
        public void EvaluateModifiedKNN(DistanceChoice distance_choice, Voting voting)
        {
            Console.WriteLine("Evalulating Modified KNN...");
            Results results = new Results(0);

            KNN knn = new KNN();
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();

            Data[] test_data = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TEST);
            string[] ingrNames = dm.GetFeatures();

            IGrouping<int, Data>[] recipes = test_data.GroupBy(d => d.recipeId).ToArray();

            int[] new_ks = new int[recipes.Length];
            int index = 0;

            // iterate through all test recipes
            foreach (IGrouping<int, Data> recipe in recipes)
            {
                // current recipe
                int[] current_recipe = dm.GetRecipe(recipe.ToArray());

                // get recommendations
                Recommendation[] recommendations = knn.GetRecommendations_ModifiedKNN(current_recipe, distance_choice, voting, ref new_ks[index]);
                index++;

                results = GetResults(results, recommendations, current_recipe);
            }

            Console.WriteLine("\nMin k: " + new_ks.Min());
            Console.WriteLine("Max k: " + new_ks.Max());
            Console.WriteLine("Avg k: " + new_ks.Average() + "\n");

            results.ShowResults();
        }
        // find optimal k
        public void GetOptimalK(DistanceChoice distance_choice, Voting voting, int max_k)
        {
            Console.WriteLine("Determining optimal k for " + distance_choice.ToString() + " distance");
            Console.WriteLine(DateTime.Now.ToLongTimeString());

            KNN knn = new KNN();
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();

            // get training data
            IDataView train_dataView = dm.GetDataView(ModelChoice.KNN, ml, DataPurpose.TRAIN);
            // get features
            IDataView features = dm.GetDataView(ModelChoice.KNN, ml, DataPurpose.FEATURES);
            string[] ingrNames = features.GetColumn<string>(features.Schema["ingrName"]).ToArray();
            // set number of folds to 5
            int num_folds = 5;

            Console.WriteLine(num_folds + "-fold cross validation...");
            // Cross validation split
            var folds = ml.Data.CrossValidationSplit(train_dataView, num_folds, samplingKeyColumnName: "recipeId");

            // keep track of f1 scores for each value of k
            double[] f1s = new double[max_k];
            // try different values of k
            for (int k = 1; k <= max_k; k++)
            {
                // show progress
                Console.WriteLine("\nk = " + k + "\t" + DateTime.Now.ToLongTimeString());

                f1s[(k - 1)] = 0.0;
                // keep track of fold results (update TP, TN, FP, FN to later determine f1 score)
                Results[] fold_results = new Results[num_folds];

                // iterate through each fold
                for (int fold = 0; fold < num_folds; fold++)
                {
                    // get training data for current fold
                    Data[] train_data = dm.GetData(folds[fold].TrainSet, features);
                    // get test data for current fold
                    Data[] validation_data = dm.GetData(folds[fold].TestSet, features);
                    // number of training recipes for current fold
                    int num_recipes = train_data.GroupBy(d => d.recipeId).ToArray().Length;
                    // number of test recipes for current fold
                    int num_validation_recipes = validation_data.GroupBy(d => d.recipeId).ToArray().Length;
                    // group test recipes by recipeId
                    IGrouping<int, Data>[] recipes = validation_data.GroupBy(d => d.recipeId).ToArray();

                    // iterate through test recipes for current fold
                    foreach (IGrouping<int, Data> current in recipes)
                    {
                        // current recipe
                        int[] recipe = dm.GetRecipe(current.ToArray());

                        // calculate distances between test recipe and training recipes, and get sorted neighbors
                        Neighbors[] distances = knn.GetDistances(distance_choice, recipe, train_data, voting);

                        // iterate through all features (unique ingredients)
                        for (int i = 0; i < ingrNames.Length; i++)
                        {
                            // keep track of votes (either recommended or not_recommended)
                            double recommended = 0;
                            double not_recommended = 0;

                            // find k nearest neighbors
                            for (int top = 0; top < k; top++)
                            {
                                // recommend ingredient if the majority of neighbors contains the ingredient
                                if (distances[top].recipe.Contains(i))
                                {
                                    if (voting.Equals(Voting.Unweighted))
                                    {
                                        recommended++;
                                    }
                                    else
                                    {
                                        recommended += distances[top].distance;
                                    }
                                }
                                else
                                {
                                    if (voting.Equals(Voting.Unweighted))
                                    {
                                        not_recommended++;
                                    }
                                    else
                                    {
                                        not_recommended += distances[top].distance;
                                    }
                                }
                            }
                            // update results for current fold
                            fold_results[fold] = UpdateResults(fold_results[fold], recipe.Contains(i), recommended >= not_recommended);
                        }
                    }
                }
                f1s[(k - 1)] = fold_results.Average(a => a.getF1());
                Console.WriteLine("Average f1: " + f1s[(k - 1)]);
            }
            // display the optimal k
            Console.WriteLine("\nOPTIMAL k = " + (Array.IndexOf(f1s, f1s.Max()) + 1) + " with an f1 score of " + f1s.Max());
        }
        // Update results (TP, TN, FP, FN) for KNN
        // ingredients are recommended if the majority of neighbors contain the ingredient
        static Results GetKNNResults(int k, Data[] test_data, string[] ingrNames, DistanceChoice distance_choice, Data[] train_data, Voting voting)
        {
            DataManager dm = new DataManager();
            KNN knn = new KNN();
            // keep track of results
            Results results = new Results(0);

            // group test data by recipeId
            IGrouping<int, Data>[] recipes = test_data.GroupBy(d => d.recipeId).ToArray();
            int count = 0;

            // iterate through all test recipes
            foreach (IGrouping<int, Data> recipe in recipes)
            {
                count++;
                // current test recipe
                int[] current_recipe = dm.GetRecipe(recipe.ToArray());

                // calculate all distances, sort neighbors by distance to current recipe
                Neighbors[] distances = knn.GetDistances(distance_choice, current_recipe, train_data, voting);

                // iterate through all features (unique ingredients)
                for (int i = 0; i < ingrNames.Length; i++)
                {
                    // keep track of votes from neighboring recipes
                    double recommended = 0;
                    double not_recommended = 0;

                    // k nearest neighbors vote
                    for (int top = 0; top < k; top++)
                    {
                        // recommend ingredient
                        if (distances[top].recipe.Contains(i))
                        {
                            if (voting.Equals(Voting.Unweighted))
                            {
                                recommended++;
                            }
                            else
                            {
                                recommended += distances[top].distance;
                            }
                        }
                        // do not recommend
                        else
                        {
                            if (voting.Equals(Voting.Unweighted))
                            {
                                not_recommended++;
                            }
                            else
                            {
                                not_recommended += distances[top].distance;
                            }
                        }
                    }
                    results = UpdateResults(results, current_recipe.Contains(i), recommended >= not_recommended);
                }
            }
            return results;
        }
        // Evaluate K Nearest Neigbors
        public void EvaluateKNN(int k, DistanceChoice distance_choice, Voting voting)
        {
            Console.WriteLine("Evaluating KNN, k = " + k + " using " + distance_choice.ToString() + " distance (" + voting.ToString() + ")...");
            Console.WriteLine(DateTime.Now.ToLongTimeString());

            DataManager dm = new DataManager();

            // get training recipes
            Data[] train_data = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TRAIN);
            // get test recipes
            Data[] test_data = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TEST);
            string[] ingrNames = dm.GetFeatures();

            // evaluate KNN and get results
            Results results = GetKNNResults(k, test_data, ingrNames, distance_choice, train_data, voting);

            // display results
            results.ShowResults();
            Console.WriteLine(DateTime.Now.ToLongTimeString());
            Console.WriteLine();
        }
        // Evaluate Naive Bayes
        public void EvaluateNB(double[][] model, bool laplace, bool normalize, bool prior)
        {
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();
            NaiveBayes nb = new NaiveBayes();

            // get test data
            IDataView testData = dm.GetDataView(ModelChoice.NB, ml, DataPurpose.TEST);
            // get features
            IDataView features = dm.GetDataView(ModelChoice.NB, ml, DataPurpose.FEATURES);

            Results results = new Results(0);

            // get distinct recipes
            int[] recipes = testData.GetColumn<int>(testData.Schema["recipeId"]).ToArray().Distinct().ToArray();
            Data[] data = dm.GetData(testData, features);

            Console.Write("\nEvaluating Naive Bayes");
            if (laplace == true)
            {
                Console.Write(" with laplace smoothing");
                if (normalize == true)
                {
                    Console.Write(" and normalization");
                }
                if (prior == false)
                {
                    Console.Write(" and uniform prior");
                }
            }
            else if (normalize == true)
            {
                Console.Write(" with normalization");
                if (prior == false)
                {
                    Console.Write(" and uniform prior");
                }
            }
            else if (prior == false)
            {
                Console.Write(" with uniform prior");
            }
            Console.WriteLine("...");

            // iterate through all recipes
            for (int r = 0; r < recipes.Length; r++)
            {
                // current recipe
                Data[] current_recipe = data.Where(d => d.recipeId == recipes[r] && d.score == 1).ToArray();
                int[] recipe = dm.GetRecipe(current_recipe);

                // get likelihoods of all possible ingredients
                Recommendation[] recommendations = nb.RecipeRecommendations(model, recipe, laplace, normalize, prior);

                // get results
                results = GetResults(results, recommendations, recipe);

            }
            // Display results
            results.ShowResults();
            Console.WriteLine();
        }

        // Evaluate Non-negative Matrix Factorization
        public void EvaluateNMF(ITransformer model)
        {
            Console.WriteLine("\nEvaluating NMF...");
            MLContext mlContext = new MLContext();

            // get test data
            DataManager dm = new DataManager();
            // test data
            IDataView testData = dm.GetDataView(ModelChoice.NMF, mlContext, DataPurpose.TEST);
            Data[] test_data = dm.GetRecipes(ModelChoice.NMF, DataPurpose.TEST);
            // train data
            IDataView trainData = dm.GetDataView(ModelChoice.NMF, mlContext, DataPurpose.TRAIN);
            Data[] train_data = dm.GetRecipes(ModelChoice.NMF, DataPurpose.TRAIN);

            // features
            string[] features = dm.GetFeatures();
            int[] recipeArray = testData.GetColumn<int>(testData.Schema["recipeId"]).ToArray();

            Results results = new Results(0);
            Recommender recommender = new Recommender();

            // distinct test recipes
            int[] distinct_recipes = recipeArray.Distinct().ToArray();

            // for each test recipe
            foreach (int r in distinct_recipes)
            {
                Recommendation[] recommendations = new Recommendation[features.Length];
                Data[] recipe = test_data.Where(d => d.recipeId == r && d.score == 1).ToArray();
                Data[] trecipe = train_data.Where(d => d.recipeId == r && d.score == 1).ToArray();
                // get recipe r
                Data[] combined = recipe.Concat(trecipe).ToArray();
                int[] current_recipe = dm.GetRecipe(combined.ToArray());

                // iterate through all features
                for (int i = 0; i < dm.GetFeatures().Length; i++)
                {
                    // make prediction (get score)
                    double prediction = recommender.SinglePrediction(mlContext, model, i, r);
                    // save score of ingredient
                    recommendations[i] = new Recommendation(new Ingredient(i, features[i]), prediction);
                }
                // sort
                recommendations = recommendations.OrderByDescending(d => d.score).ToArray();
                results = GetResults(results, recommendations, current_recipe);
            }
            // Display accuracy results
            results.ShowResults();
            Console.WriteLine();
        }
        // top recommendation 
        // return updated results
        public Results GetResults(Results results, Recommendation[] recommendations, int[] recipe)
        {
            // iterate through all possible ingredients and calculate their likelihoods
            for (int i = 0; i < recommendations.Length; i++)
            {
                // pick top recommendation
                Recommendation recommendation = recommendations[0];

                // get top recommendation, discarding other ingredients also in the recipe
                for (int j = 0; j < recipe.Length && recipe.Contains(recommendations[j].ingredient.id); j++)
                {
                    recommendation = recommendations[j];
                    if (recommendations[j].ingredient.id == i)
                    {
                        break;
                    }
                }
                results = UpdateResults(results, recipe.Contains(i), recommendation.ingredient.id == i);
            }

            return results;
        }
        static Results UpdateResults(Results results, bool actual, bool predicted)
        {
            // Actual = Yes (ingredient belongs in recipe)
            if (actual == true)
            {
                // Predicted = Yes
                if (predicted == true)
                {
                    // True Positive
                    results.TP++;
                }
                // Predicted = No
                else
                {
                    // False Negative
                    results.FN++;
                }
            }
            // Actual = No
            else
            {
                // Predicted = Yes
                if (predicted == true)
                {
                    results.FP++;
                }
                // Predicted = No
                else
                {
                    // True Negative
                    results.TN++;
                }
            }
            return results;
        }
    }
}
