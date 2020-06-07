using System;
using System.Runtime.Serialization;

namespace IngredientRecommender
{
    class Program
    {
        static void Main(string[] args)
        {
            bool exit = false;
            while (exit == false)
            {
                exit = run();
            }
        }
        static bool run()
        {
            Console.WriteLine("\nSelect an option below:");
            Console.WriteLine("0: Exit");
            Console.WriteLine("1: Evaluate models");
            Console.WriteLine("2: Get Recommendations");
            Console.WriteLine("3: See example recipes");
            Console.WriteLine("4: See all ingredients");
            Console.WriteLine("5: Find optimal k");

            string option = Console.ReadLine();
            UserRequest req = new UserRequest();

            // Evaluate models
            if (option == "1")
            {
                Console.WriteLine("\nWhich model do you want to evaluate?");
                Console.WriteLine("1: Naive Bayes");
                Console.WriteLine("2: Non-negative Matrix Factorization (12 minutes)");
                Console.WriteLine("3: k Nearest Neighbor");
                Console.WriteLine("4: Modified k Nearest Neighbor");

                option = Console.ReadLine();
                if (option == "1")
                {
                    Evaluate(ModelChoice.NB);
                }
                else if (option == "2")
                {
                    Evaluate(ModelChoice.NMF);
                }
                else if (option == "3")
                {
                    Evaluate(ModelChoice.KNN);
                }
                else if (option == "4")
                {
                    Evaluate(ModelChoice.MKNN);
                }
                else
                {
                    Console.WriteLine("none of the options were selected.");
                }

                return false;
            }
            // Get Recommendations
            else if (option == "2")
            {
                Console.WriteLine("Type in a recipe (comma delimited)");
                string recipe_response = Console.ReadLine();
                string[] ingrs = recipe_response.Split(',');

                Console.WriteLine("\nChoose one of the following options:");
                Console.WriteLine("1: Recommend ingredients to ADD");
                Console.WriteLine("2: Recommend ingredients to REMOVE");

                option = Console.ReadLine();
                UserRequest request = new UserRequest();

                // Recommend ingredients to add to recipe
                if (option == "1")
                {
                    request.TopRecommendations(20, ingrs, ModelChoice.NB, true, false);
                }
                // Recommend ingredients to remove from recipe
                else if (option == "2")
                {
                    request.TopRecommendations(20, ingrs, ModelChoice.NB, false, true);
                }
                return false;
            }
            // See example recipes
            else if (option == "3")
            {
                req.SeeExampleRecipes();
                return false;
            }
            // See all ingredients
            else if (option == "4")
            {
                req.GetAllIngredients(true);
                return false;
            }
            // Get optimal k
            else if (option == "5")
            {
                Console.WriteLine("Which distance measure?");
                Console.WriteLine("1: Jaccard");
                Console.WriteLine("2: Hamming");
                Console.WriteLine("3: Levenshtein");

                option = Console.ReadLine();
                Evaluation eval = new Evaluation();

                if (option == "1")
                {
                    eval.GetOptimalK(DistanceChoice.Jaccard, Voting.Unweighted, 40);
                }
                else if (option == "2")
                {
                    eval.GetOptimalK(DistanceChoice.Hamming, Voting.Unweighted, 40);
                }
                else if (option == "3")
                {
                    eval.GetOptimalK(DistanceChoice.Levenshtein, Voting.Unweighted, 40);
                }
                return false;
            }
            return true;
        }
        // Evaluate different models
        static void Evaluate(ModelChoice choice)
        {
            Evaluation eval = new Evaluation();

            // Evaluate Non-negative Matrix Factorization (12 minutes)
            if (choice.Equals(ModelChoice.NMF))
            {
                Recommender nmf = new Recommender();
                eval.EvaluateNMF(nmf.GetModel());
            }
            // Evaluate Naive Bayes (5 seconds each)
            else if (choice.Equals(ModelChoice.NB))
            {
                NaiveBayes nb = new NaiveBayes();
                double[][] model = nb.GetModel();

                Console.WriteLine("1: Evaluate ALL versions of Naive Bayes");
                Console.WriteLine("2: Evaluate only the BEST version of Naive Bayes (normalization, laplace, uniform prior)");

                string option = Console.ReadLine();
                // ALL
                if (option == "1")
                {
                    eval.EvaluateNB(model, false, false, true);
                    eval.EvaluateNB(model, true, false, true); // with laplace smoothing
                    eval.EvaluateNB(model, false, true, true); // with normalization (best results)
                    eval.EvaluateNB(model, true, true, true);  // with laplace smoothing and normalization

                    // uniform prior
                    eval.EvaluateNB(model, false, false, false);
                    eval.EvaluateNB(model, true, false, false); // with laplace smoothing
                    eval.EvaluateNB(model, false, true, false); // with normalization (best results)
                }
                // BEST 
                eval.EvaluateNB(model, true, true, false);  // with laplace smoothing, normalization, and uniform prior
            }
            // Evaluate KNN (35 seconds)
            else if (choice.Equals(ModelChoice.KNN))
            {
                eval.EvaluateKNN(6, DistanceChoice.Jaccard, Voting.Unweighted);
            }
            // Evaluate Modified KNN (35 seconds)
            else
            {
                eval.EvaluateModifiedKNN(DistanceChoice.Jaccard_Similarity, Voting.Unweighted);
            }
        }
        static void CompareRecommendations()
        {
            UserRequest request = new UserRequest();
            // input recipe examples
            string[] first = new string[2] { "apple", "sugar" };
            string[] second = new string[4] { "bacon", "chicken", "salt", "black pepper" };
            string[] third = new string[4] { "soy sauce", "ginger", "chicken", "water" };
            string[] fourth = new string[4] { "mushrooms", "chicken breast", "chicken broth", "sugar" };
            string[] fifth = new string[7] { "confectioners sugar", "cocoa powder", "condensed milk", "applesauce", "almond extract", "cheddar cheese", "buttermilk" };

            foreach (ModelChoice item in Enum.GetValues(typeof(ModelChoice)))
            {
                // recommend ingredients to ADD
                request.TopRecommendations(10, first, item, true, false);
                request.TopRecommendations(10, second, item, true, false);
                request.TopRecommendations(10, third, item, true, false);
                // recommend ingredients to REMOVE
                request.TopRecommendations(10, fourth, item, false, false);
                request.TopRecommendations(10, fifth, item, false, false);
            }
        }
    }
}
