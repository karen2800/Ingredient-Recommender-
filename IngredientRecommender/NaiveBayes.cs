using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace IngredientRecommender
{
    class NaiveBayes
    {
        // Get recommendations for a recipe
        // the ingredient score is equivalent to the posterior
        public Recommendation[] RecipeRecommendations(double[][] model, int[] recipe, bool laplace, bool normalize, bool prior)
        {
            MLContext ml = new MLContext();
            DataManager dm = new DataManager();

            // get features (unique ingredients)
            string[] ingr_names = dm.GetFeatures();

            // get number of training recipes
            IDataView allrecipes = dm.GetDataView(ModelChoice.NB, ml, DataPurpose.TRAIN);
            int num_recipes = allrecipes.GetColumn<int>(allrecipes.Schema["recipeId"]).ToArray().Length;

            Recommendation[] recommendations = new Recommendation[ingr_names.Length];

            // iterate through all features (unique ingredients)
            for (int f = 0; f < ingr_names.Length; f++)
            {
                double likelihood = 1.0;
                // iterate through all the ingredients in the recipe
                foreach (int i in recipe)
                {
                    // ignore matching ingredients
                    if (i != f)
                    {
                        // laplace smoothing
                        if (laplace == true)
                        {
                            likelihood *= (model[i][f] + 1.0) / (model[f][f] + ingr_names.Length);
                        }
                        else
                        {
                            likelihood *= model[i][f] / model[f][f];
                        }

                        // normalize
                        if (normalize == true)
                        {
                            likelihood /= model[i][i];
                        }
                    }
                }
                // prior
                if (prior == true)
                {
                    likelihood *= model[f][f] / num_recipes;
                }
                recommendations[f] = new Recommendation(new Ingredient(f, ingr_names[f]), likelihood);
            }
            // sort
            recommendations = recommendations.OrderByDescending(t => t.score).ToArray();

            return recommendations;
        }
        // Build Model
        // return matrix of co-occurence counts between ingredients (number of recipes both ingredients are seen together)
        public double[][] GetModel()
        {
            MLContext mlContext = new MLContext();
            DataManager dm = new DataManager();

            // get training data
            Data[] train_recipes = dm.GetRecipes(ModelChoice.NB, DataPurpose.TRAIN);
            IGrouping<int, Data>[] distinct_recipes = train_recipes.Where(r => r.score == 1).GroupBy(r => r.recipeId).ToArray();
            // get features (ingredients)
            string[] ingrNames = dm.GetFeatures();

            // matrix of ingredients
            double[][] model = new double[ingrNames.Length][];

            // iterate through all recipes
            foreach (IGrouping<int, Data> recipe in distinct_recipes)
            {
                // get recipe
                Data[] currRecipe = recipe.ToArray();

                // iterate through ingredients of the current recipe
                foreach (Data i in currRecipe)
                {
                    foreach (Data j in currRecipe)
                    {
                        if (model[i.ingredient.id] == null)
                        {
                            model[i.ingredient.id] = new double[ingrNames.Length];
                        }
                        // update co-occurrence count
                        model[i.ingredient.id][j.ingredient.id]++;
                    }
                }
            }
            return model;
        }
    }
}
