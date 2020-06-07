using System;
using System.Collections.Generic;
using System.Text;

using Microsoft.ML;
using Microsoft.ML.Trainers;
using System.Linq;

namespace IngredientRecommender
{
    class Recommender
    {
        // Single Prediction
        public double SinglePrediction(MLContext mLContext, ITransformer model, int ingredient, int recipe)
        {
            var predictionengine = mLContext.Model.CreatePredictionEngine<IngredientRating, IngredientRatingPrediction>(model);
            var prediction = predictionengine.Predict(
                new IngredientRating()
                {
                    ingrId = ingredient,
                    recipeId = recipe
                });

            return prediction.Score;
        }
        // Get Model
        public ITransformer GetModel()
        {
            MLContext mlContext = new MLContext();

            DataManager dm = new DataManager();
            IDataView trainData = dm.GetDataView(ModelChoice.NMF, mlContext, DataPurpose.TRAIN);

            // Build and Train Model
            ITransformer model = BuildTrain(mlContext, trainData);

            return model;
        }

        // Build and Train Model
        static ITransformer BuildTrain(MLContext mlContext, IDataView data)
        {

            IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "recipeIdEncoded", inputColumnName: "recipeId")
                .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "ingredientIdEncoded", inputColumnName: "ingrId"));

            // set Matrix factorization options
            var options = new MatrixFactorizationTrainer.Options
            {
                MatrixColumnIndexColumnName = "recipeIdEncoded",
                MatrixRowIndexColumnName = "ingredientIdEncoded",
                LabelColumnName = "Label",
                NumberOfIterations = 150,
                ApproximationRank = 100,
                Lambda = 0.025,
                LearningRate = 0.1,
                NonNegative = true,
                LossFunction = MatrixFactorizationTrainer.LossFunctionType.SquareLossRegression
            };

            var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

            Console.WriteLine("=============== Training the model ===============");

            // train model
            ITransformer model = trainerEstimator.Fit(data);

            return model;
        }
    }
}
