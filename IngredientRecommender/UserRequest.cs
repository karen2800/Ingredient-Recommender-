using System;
using System.Collections.Generic;
using System.Text;
using System.Linq;
using Microsoft.ML;

namespace IngredientRecommender
{
    class UserRequest
    {
        public void SeeExampleRecipes()
        {
            GetRecipeWithIngr("chicken");
            GetRecipeWithIngr("basil");
            GetRecipeWithIngr("sugar");
            GetRecipeWithIngr("zucchini");
            GetRecipeWithIngr("apple");
            GetRecipeWithIngr("butter");
            GetRecipeWithIngr("worcestershire sauce");
            GetRecipeWithIngr("potato");
            GetRecipeWithIngr("banana");
        }
        public void GetStats()
        {
            DataManager dm = new DataManager();
            // get training data
            Data[] train_data = dm.GetRecipes(ModelChoice.NB, DataPurpose.TRAIN);
            // get test data
            Data[] test_data = dm.GetRecipes(ModelChoice.NB, DataPurpose.TEST);
            // total number of recipes
            int total_recipes = train_data.GroupBy(d => d.recipeId).ToArray().Length + test_data.GroupBy(d => d.recipeId).ToArray().Length;
            Console.WriteLine(total_recipes.ToString() + " recipes");
            // get features
            string[] features = dm.GetFeatures();
            Console.WriteLine(features.Length + " ingredients");
        }
        // get recipe containing a specific ingredient
        public void GetRecipeWithIngr(string ingr)
        {
            DataManager dm = new DataManager();
            Data[] data = dm.GetRecipes(ModelChoice.NB, DataPurpose.TRAIN);
            Console.WriteLine("\nRecipe containing: " + ingr);
            int r = data.Where(d => d.ingredient.name == ingr && d.score == 1).ToArray()[0].recipeId;
            Data[] recipe = GetRecipe(r, true);
        }
        // get recipe
        // return Data[]
        public Data[] GetRecipe(int r, bool display_recipe)
        {
            DataManager dm = new DataManager();
            Data[] data = dm.GetRecipes(ModelChoice.NB, DataPurpose.TRAIN);

            // random recipe
            if (r == -1)
            {
                Random rand = new Random();
                r = rand.Next(0, data.GroupBy(d => d.recipeId).ToArray().Length);
            }

            // get recipe
            Data[] recipe = data.Where(d => d.recipeId == r && d.score == 1).ToArray();

            // display recipe
            if (display_recipe == true)
            {
                Console.Write(recipe[0].ingredient.name);
                for (int i = 1; i < recipe.Length; i++)
                {
                    Console.Write(", " + recipe[i].ingredient.name);
                }
                Console.WriteLine();
            }
            return recipe;
        }
        // Get all ingredients
        public string[] GetAllIngredients(bool display)
        {
            DataManager dm = new DataManager();
            string[] features = dm.GetFeatures();
            if (display == true)
            {
                Console.WriteLine();
                for (int i = 0; i < features.Length; i++)
                {
                    Console.WriteLine(features[i]);
                }
                Console.WriteLine();
            }
            return features;
        }
        static void PrintRecipe(string[] recipe)
        {
            Console.Write(recipe[0]);
            for (int i = 1; i < recipe.Length; i++)
            {
                Console.Write(", " + recipe[i]);
            }
            Console.WriteLine();
        }
        static void PrintRecipe(int[] recipe)
        {
            DataManager dm = new DataManager();
            string[] features = dm.GetFeatures();

            Console.Write(features[recipe[0]]);
            for (int i = 1; i < recipe.Length; i++)
            {
                Console.Write(", " + features[recipe[i]]);
            }
            Console.WriteLine();
        }
        // Ingredients to ADD to or REMOVE from a recipe
        public void TopRecommendations(int top, string[] recipe_str, ModelChoice model_choice, bool add, bool include_recipe_ingrs)
        {
            DataManager dm = new DataManager();
            // get training data
            Data[] data = dm.GetRecipes(model_choice, DataPurpose.TRAIN);

            Console.WriteLine("You model choice: " + model_choice.ToString());

            // input recipe
            int[] recipe = new int[recipe_str.Length];
            for (int i = 0; i < recipe_str.Length; i++)
            {
                try
                {
                    // trim and make lowercase
                    recipe_str[i] = recipe_str[i].Trim().ToLower();

                    // find ingredient 
                    recipe[i] = data.Where(d => d.ingredient.name.Equals(recipe_str[i])).ToArray()[0].ingredient.id;
                }
                catch
                {
                    // get features (ingredients)
                    string[] features = GetAllIngredients(false);

                    bool found = false;
                    // try finding a similar ingredient
                    foreach (string ingr in features)
                    {
                        if (ingr.StartsWith(recipe_str[i]) || ingr.Contains(recipe_str[i]))
                        {
                            recipe_str[i] = ingr;
                            recipe[i] = data.Where(d => d.ingredient.name.Equals(ingr)).ToArray()[0].ingredient.id;
                            found = true;
                            break;
                        }
                    }
                    // ingredient not found
                    if (found == false)
                    {
                        Console.WriteLine("Ingredient [" + recipe_str[i] + "] was not found");
                        return;
                    }
                }
            }

            Console.Write("Your recipe: ");
            PrintRecipe(recipe_str);

            // keep track of ingredient recommendations
            Recommendation[] recommendations = null;
            // Naive Bayes
            if (model_choice.Equals(ModelChoice.NB))
            {
                NaiveBayes nb = new NaiveBayes();
                recommendations = nb.RecipeRecommendations(nb.GetModel(), recipe, true, true, false);
            }
            // k Nearest Neighbors
            else if (model_choice.Equals(ModelChoice.KNN))
            {
                KNN knn = new KNN();
                recommendations = knn.GetRecommendations(6, DistanceChoice.Jaccard, recipe, Voting.Unweighted);
            }
            // Modified k Nearest Neighbors (dynamic k)
            else if (model_choice.Equals(ModelChoice.MKNN))
            {
                KNN knn = new KNN();
                int new_k = 0;
                recommendations = knn.GetRecommendations_ModifiedKNN(recipe, DistanceChoice.Jaccard_Similarity, Voting.Unweighted, ref new_k);
            }
            else
            {
                return;
            }
            // Ingredients to Add
            if (add == true)
            {
                Console.WriteLine("Your recommendations:");

                for (int i = 0; i < top; i++)
                {
                    // skip ingredients in recipe
                    if (include_recipe_ingrs == false && recipe_str.Contains(recommendations[i].ingredient.name))
                    {
                        top++;
                        continue;
                    }
                    Console.WriteLine(recommendations[i].ingredient.name);
                }
            }
            // Ingredients to Remove
            else
            {
                Console.WriteLine("Ingredients ordered by what to remove first:");
                // only keep scores of ingredients in input recipe
                recommendations = recommendations.Where(d => recipe.Contains(d.ingredient.id)).ToArray();
                // sort by score
                recommendations = recommendations.OrderBy(d => d.score).ToArray();
                for (int i = 0; i < recipe.Length; i++)
                {
                    Console.WriteLine(recommendations[i].ingredient.name);
                }
            }
            Console.WriteLine();
        }
        // Get ingredients sorted by what is most recommended
        public Recommendation[] GetRecommendations(ModelChoice model_choice, int[] recipe)
        {
            Recommendation[] recommendations = null;

            if (model_choice.Equals(ModelChoice.NB))
            {
                NaiveBayes nb = new NaiveBayes();
                recommendations = nb.RecipeRecommendations(nb.GetModel(), recipe, true, true, false);
            }
            else if (model_choice.Equals(ModelChoice.KNN))
            {
                KNN knn = new KNN();
                recommendations = knn.GetRecommendations(6, DistanceChoice.Jaccard, recipe, Voting.Unweighted);
            }
            else if (model_choice.Equals(ModelChoice.MKNN))
            {
                KNN knn = new KNN();
                int new_k = 0;
                recommendations = knn.GetRecommendations_ModifiedKNN(recipe, DistanceChoice.Jaccard_Similarity, Voting.Unweighted, ref new_k);
            }

            return recommendations;
        }
    }
}
