# Diabetes Regression Project

## Purpose of the project
This project uses the built-in diabetes dataset from scikit-learn to test how well different machine learning regression models can predict disease progression. The goal is to compare three models using the same train/test split so their performance can be judged fairly. This gives a simple but meaningful example of how machine learning can support medical prediction tasks.

## Models included
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor

## Class design and implementation
The project is built around one main class called `DiabetesRegressionProject`. I used a class so the full workflow could stay organized in one place instead of spreading the steps across random code blocks. The class handles loading the dataset, splitting the data into training and testing sets, training the models, evaluating the models, and saving the final results.

### Class attributes
- `test_size`: controls how much of the dataset is reserved for testing.
- `random_state`: keeps the split and model results reproducible.
- `output_dir`: folder where result files are saved.
- `data`: stores the feature values from the diabetes dataset.
- `target`: stores the regression target values.
- `X_train`, `X_test`, `y_train`, `y_test`: store the training and testing data after the split.
- `models`: dictionary containing the three initialized regression models.

## Methods
- `load_data()`: loads the built-in diabetes dataset into pandas objects.
- `split_data()`: performs the standard train/test split.
- `train_and_evaluate()`: trains each regression model and calculates MAE, RMSE, and R².
- `save_results()`: saves the model comparison table as CSV and JSON files.
- `run()`: runs the full workflow from beginning to end.

## Evaluation metrics used
- **MAE (Mean Absolute Error):** shows the average prediction error in absolute terms.
- **RMSE (Root Mean Squared Error):** gives more weight to larger prediction mistakes.
- **R² (Coefficient of Determination):** shows how much of the variance in the target is explained by the model.

## Limitations
This project uses a small built-in dataset, so results may not fully reflect real clinical settings. The models are compared using one train/test split instead of a more advanced cross-validation process. Also, only basic model settings are used, so performance could likely improve with deeper hyperparameter tuning.

## Files produced
Running the Python script creates:
- `diabetes_model_results.csv`
- `diabetes_model_results.json`

These files make it easy to review or submit the model comparison results.
