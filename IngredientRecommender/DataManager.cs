using System;
using System.Collections.Generic;
using System.Text;

using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.IO;

namespace IngredientRecommender
{
    enum ModelChoice
    {
        NB, // Naive Bayes
        NMF, // Non-negative Matrix Factorization
        KNN, // K-Nearest Neighbors
        MKNN // Modified K-Nearest Neighbors
    }
    enum DataPurpose
    {
        TRAIN,
        TEST,
        FEATURES
    }
    public class IngredientRating
    {
        [LoadColumn(0)]
        public int recipeId;
        [LoadColumn(1)]
        public int ingrId;
        [LoadColumn(2)]
        public float Label;
    }
    public class IngredientRatingPrediction
    {
        public float Label;
        public float Score;
    }
    public class IngredientFeatures
    {
        [LoadColumn(0)]
        public int ingrId;
        [LoadColumn(1)]
        public string ingrName;
    }
    struct Ingredient
    {
        public int id;
        public string name;

        public Ingredient(int i, string n)
        {
            id = i;
            name = n;
        }
    }
    struct Data
    {
        public int recipeId;
        public Ingredient ingredient;
        public int score;
        public Data(int r, Ingredient i, int s)
        {
            recipeId = r;
            ingredient = i;
            score = s;
        }
    }
    struct Recommendation
    {
        public Ingredient ingredient;
        public double score;

        public Recommendation(Ingredient i, double l)
        {
            ingredient = i;
            score = l;
        }
    }
    class DataManager
    {
        public static string TRAIN_DATA = "nb-train-data.csv";
        public static string TEST_DATA = "nb-test-data.csv";
        public static string NMF_TRAIN_DATA = "train-data.csv";
        public static string NMF_TEST_DATA = "test-data.csv";
        public static string ALL_DATA = "all-data.csv";
        public static string FEATURES_DATA = "features-data.csv";

        // Load Data
        public IDataView LoadData(string path, MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<IngredientRating>(path,
                hasHeader: true,
                separatorChar: ',');

            return dataView;
        }
        // Load Features
        public IDataView LoadFeatures(string path, MLContext mlContext)
        {
            IDataView dataView = mlContext.Data.LoadFromTextFile<IngredientFeatures>(path,
                hasHeader: true,
                separatorChar: ',');

            return dataView;
        }
        // Get DataView
        public IDataView GetDataView(ModelChoice choice, MLContext ml, DataPurpose purpose)
        {
            // get features
            if (purpose.Equals(DataPurpose.FEATURES))
            {
                return LoadFeatures(Path.Combine(Environment.CurrentDirectory, "Data", FEATURES_DATA), ml);
            }
            // get NMF data (entries set aside for testing)
            else if (choice.Equals(ModelChoice.NMF))
            {
                // test data
                if (purpose.Equals(DataPurpose.TEST))
                {
                    return LoadData(Path.Combine(Environment.CurrentDirectory, "Data", NMF_TEST_DATA), ml);
                }
                // training data
                else
                {
                    return LoadData(Path.Combine(Environment.CurrentDirectory, "Data", NMF_TRAIN_DATA), ml);
                }
            }
            // get data (recipes set aside for testing)
            else
            {
                // test data
                if (purpose.Equals(DataPurpose.TEST))
                {
                    return LoadData(Path.Combine(Environment.CurrentDirectory, "Data", TEST_DATA), ml);
                }
                // training data
                else
                {
                    return LoadData(Path.Combine(Environment.CurrentDirectory, "Data", TRAIN_DATA), ml);
                }
            }
        }
        // Get Data
        // row# : recipeId, Ingredient (id, name), score (1=present, 0=NOT_present in recipe)
        public Data[] GetData(IDataView dataView, IDataView features)
        {
            int[] ingredients = dataView.GetColumn<int>(dataView.Schema["ingrId"]).ToArray();
            int[] recipes = dataView.GetColumn<int>(dataView.Schema["recipeId"]).ToArray();
            float[] scores = dataView.GetColumn<float>(dataView.Schema["Label"]).ToArray();
            string[] ingrNames = features.GetColumn<string>(features.Schema["ingrName"]).ToArray();

            Data[] data = new Data[recipes.Length];

            for (int row = 0; row < recipes.Length; row++)
            {
                data[row] = new Data(recipes[row], new Ingredient(ingredients[row], ingrNames[ingredients[row]]), (int)scores[row]);
            }

            return data;
        }
        // Get Recipe
        // Convert Data[] to int[]
        public int[] GetRecipe(Data[] currRecipe)
        {
            int[] recipeIngrs = new int[currRecipe.Length];
            for (int i = 0; i < currRecipe.Length; i++)
            {
                recipeIngrs[i] = currRecipe[i].ingredient.id;
            }

            return recipeIngrs;
        }
        // Get Features (ingredient names)
        public string[] GetFeatures()
        {
            MLContext ml = new MLContext();
            IDataView features_view = GetDataView(ModelChoice.NB, ml, DataPurpose.FEATURES);
            string[] features = features_view.GetColumn<string>(features_view.Schema["ingrName"]).ToArray();
            return features;
        }
        // Get Recipes
        public Data[] GetRecipes(ModelChoice model_choice, DataPurpose data_purpose)
        {
            MLContext mlContext = new MLContext();
            IDataView testData = GetDataView(model_choice, mlContext, data_purpose);
            IDataView features = GetDataView(model_choice, mlContext, DataPurpose.FEATURES);

            return GetData(testData, features);
        }
        public void GetDataStats(bool ingr_dist)
        {
            Data[] test = GetRecipes(ModelChoice.NB, DataPurpose.TEST);
            Data[] train = GetRecipes(ModelChoice.NB, DataPurpose.TRAIN);
            string[] features = GetFeatures();
            Data[] data = null;

            double avg_num_ingr = 0.0;
            int min_num_ingr = int.MaxValue;
            int max_num_ingr = int.MinValue;
            int count = 0;
            int[] num_recipes_per_ingr = new int[features.Length];

            // test and train data
            for (int i = 0; i < 2; i++)
            {
                if (i == 0)
                {
                    data = train;
                }
                else
                {
                    data = test;
                }
                // group by recipeId
                foreach (IGrouping<int, Data> recipe in data.GroupBy(d => d.recipeId).ToArray())
                {
                    int[] current_recipe = GetRecipe(recipe.ToArray());
                    // keep track of number of recipes per ingr
                    for (int j = 0; j < current_recipe.Length; j++)
                    {
                        num_recipes_per_ingr[current_recipe[j]]++;
                    }

                    if (current_recipe.Length < min_num_ingr)
                    {
                        min_num_ingr = current_recipe.Length;
                    }
                    if (current_recipe.Length > max_num_ingr)
                    {
                        max_num_ingr = current_recipe.Length;
                    }
                    avg_num_ingr += current_recipe.Length;
                    count++;
                }
            }

            Console.WriteLine("Average number of ingredients per recipe: " + (avg_num_ingr / count));
            Console.WriteLine("Min: " + min_num_ingr);
            Console.WriteLine("Max: " + max_num_ingr);

            // See Ingredient Distribution
            if (ingr_dist == true)
            {
                num_recipes_per_ingr = num_recipes_per_ingr.OrderByDescending(r => r).ToArray();
                foreach (int i in num_recipes_per_ingr)
                {
                    Console.WriteLine(i);
                }
            }
        }
    }
}
