using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML;
using System.Linq;

namespace IngredientRecommender
{
    enum DistanceChoice
    {
        Hamming,
        Levenshtein,
        Jaccard,
        Jaccard_Similarity
    }
    enum Voting
    {
        Unweighted,
        Weighted
    }
    struct Neighbors
    {
        public double distance;
        public int[] recipe;

        public Neighbors(double d, int[] r)
        {
            distance = d;
            recipe = r;
        }
    }
    class KNN
    {
        // Get Modified KNN Recommendations
        public Recommendation[] GetRecommendations_ModifiedKNN(int[] current_recipe, DistanceChoice distance_choice, Voting voting, ref int new_k)
        {
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();
            int max_k = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TEST).GroupBy(d => d.recipeId).ToArray().Length;

            Data[] train_data = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TRAIN);
            string[] ingrNames = dm.GetFeatures();

            // keep track of ingredient scores (recommendations)
            Recommendation[] recommendations = new Recommendation[ingrNames.Length];
            for (int i = 0; i < ingrNames.Length; i++)
            {
                recommendations[i] = new Recommendation(new Ingredient(i, ingrNames[i]), 0.0);
            }

            // calculate all distances, find nearest neighbors to current recipe
            Neighbors[] distances = GetDistances(distance_choice, current_recipe, train_data, voting);

            recommendations = ModifiedKNN(recommendations, distances, current_recipe, ingrNames, ref new_k, max_k);

            // sort 
            recommendations = recommendations.OrderByDescending(d => d.score).ToArray();

            return recommendations;
        }
        public Recommendation[] ModifiedKNN(Recommendation[] recommendations, Neighbors[] distances, int[] current_recipe, string[] ingrNames, ref int new_k, int max_k)
        {
            // TODO: REMOVE iterate through all features (unique ingredients)
            for (int k = 0; k < max_k; k++)
            {
                // for each ingredient in current neighbor, increase ingrs by similarity score
                foreach (int ingr in distances[k].recipe)
                {
                    recommendations[ingr].score += distances[k].distance;
                }

                // check if most ingredients in test_recipe seen in neighboring k recipes
                int seen = 0;
                foreach (int ingr in current_recipe)
                {
                    if (recommendations[ingr].score > 0)
                    {
                        seen++;
                    }
                }

                // if most ingredients seen, we have found a good k value
                // minimum value of k = 6 (optimal k of Jaccard distance)
                if (seen >= current_recipe.Length && (k + 1) >= 6)
                {
                    new_k = (k + 1);
                    break;
                }
            }
            return recommendations;
        }
        // Get recommendations
        public Recommendation[] GetRecommendations(int k, DistanceChoice distance_choice, int[] recipe, Voting voting)
        {
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();
            // get features (ingredient names)
            string[] ingrNames = dm.GetFeatures();
            // get training recipes
            Data[] data = dm.GetRecipes(ModelChoice.KNN, DataPurpose.TRAIN);

            Recommendation[] recommendations = new Recommendation[ingrNames.Length];
            // iterate through all ingredients
            for (int i = 0; i < ingrNames.Length; i++)
            {
                Ingredient current_ingr = new Ingredient(i, ingrNames[i]);
                // calculate all distances
                Neighbors[] distances = GetDistances(distance_choice, recipe, data, voting);

                double recommended = 0;
                double not_recommended = 0;

                // k nearest neighbors vote
                // recommend ingredient if the majority of neighbors contains the ingredient
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
                    // do not recommend ingredient
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
                recommendations[i] = new Recommendation(current_ingr, (recommended + 1.0) / (not_recommended + 2.0));
            }
            recommendations = recommendations.OrderByDescending(r => r.score).ToArray();
            return recommendations;
        }
        // Get Distances
        // int[] recipe_a = test recipe
        // Data[] data = training data
        public Neighbors[] GetDistances(DistanceChoice distance_choice, int[] recipe_a, Data[] data, Voting voting)
        {
            DataManager dm = new DataManager();

            // group by recipeId
            IGrouping<int, Data>[] recipes = data.GroupBy(d => d.recipeId).ToArray();
            // keep track of distances between neighboring recipes
            Neighbors[] distances = new Neighbors[recipes.Length];

            int index = 0;
            // iterate through all training recipes
            foreach (IGrouping<int, Data> recipe in recipes)
            {
                // current recipe
                Data[] this_recipe = recipe.ToArray();
                int[] current_recipe = dm.GetRecipe(recipe.ToArray());

                // Hamming distance
                if (distance_choice.Equals(DistanceChoice.Hamming))
                {
                    distances[index] = new Neighbors(HammingDistance(recipe_a, current_recipe, voting), current_recipe);
                }
                // Levenshtein distance
                else if (distance_choice.Equals(DistanceChoice.Levenshtein))
                {
                    distances[index] = new Neighbors(LevenshteinDistance(recipe_a, current_recipe, voting), current_recipe);
                }
                // Jaccard distance
                else if (distance_choice.Equals(DistanceChoice.Jaccard))
                {
                    distances[index] = new Neighbors(JaccardDistance(recipe_a, current_recipe, voting), current_recipe);
                }
                // Jaccard similarity
                else if (distance_choice.Equals(DistanceChoice.Jaccard_Similarity))
                {
                    distances[index] = new Neighbors(JaccardSimilarity(recipe_a, current_recipe, voting), current_recipe);
                }
                index++;
            }
            // sort by distance
            if (distance_choice.Equals(DistanceChoice.Jaccard_Similarity))
            {
                distances = distances.OrderByDescending(n => n.distance).ToArray();
            }
            else
            {
                distances = distances.OrderBy(n => n.distance).ToArray();
            }

            return distances;
        }
        // Hamming Distance
        static double HammingDistance(int[] a, int[] b, Voting voting)
        {
            // number of differences between two recipes
            double distance = a.Length + b.Length - (a.Intersect(b).ToArray().Length * 2);

            // Weighted voting
            if (voting.Equals(Voting.Weighted))
            {
                return 1 / (distance * distance);
            }
            // Unweighted voting
            else
            {
                return distance;
            }
        }
        // Levenshtein distance (minimum number of edits)
        static double LevenshteinDistance(int[] a, int[] b, Voting voting)
        {
            int[][] matrix = new int[a.Length + 1][];

            for (int i = 0; i < a.Length + 1; i++)
            {
                int[] curr_a = a.Take(i).ToArray();
                matrix[i] = new int[b.Length + 1];

                for (int j = 0; j < b.Length + 1; j++)
                {
                    int[] curr_b = a.Take(j).ToArray();

                    if (Math.Min(curr_a.Length, curr_b.Length) == 0)
                    {
                        matrix[i][j] = Math.Max(curr_a.Length, curr_b.Length);
                    }
                    else
                    {
                        int x = matrix[i - 1][j] + 1;
                        int y = matrix[i][j - 1] + 1;
                        int z = matrix[i - 1][j - 1];
                        if (a[i - 1] != b[j - 1])
                        {
                            z += 1;
                        }
                        matrix[i][j] = Math.Min(Math.Min(x, y), z);
                    }
                }
            }
            double distance = matrix[a.Length][b.Length];

            // Weighted voting
            if (voting.Equals(Voting.Weighted))
            {
                return 1 / (distance * distance);
            }
            // Unweighted voting
            else
            {
                return distance;
            }
        }
        // Jaccard distance
        static double JaccardDistance(int[] a, int[] b, Voting voting)
        {
            double intersect = a.Intersect(b).ToArray().Length;
            double union = a.Union(b).ToArray().Length;
            double distance = 1 - (intersect / union);

            // Weighted voting
            if (voting.Equals(Voting.Weighted))
            {
                return 1 / (distance * distance);
            }
            // Unweighted voting
            else
            {
                return distance;
            }
        }
        // Jaccard Similarity
        static double JaccardSimilarity(int[] a, int[] b, Voting voting)
        {
            double intersect = a.Intersect(b).ToArray().Length;
            double union = a.Union(b).ToArray().Length;
            double distance = (intersect / union);

            // Weighted voting
            if (voting.Equals(Voting.Weighted))
            {
                return 1 / (distance * distance);
            }
            // Unweighted voting
            else
            {
                return distance;
            }
        }
    }
}
