# =======================================================================
# CLASSES
# Define custom classes in this file.
# =======================================================================
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score

import src.functions as sf

filepath = os.path.join("datasets", "dataset_cell_growth_4factor.csv")

FEATURES = ['Temperature [C]', 'pH', 'Glucose [mM]', 'Lactate [mM]']
TARGET = 'Specific Growth Rate [d^-1]'

MODEL_NAMES = ["Polynomial", "KNN"]

# If the polynomial's validation RMSE is within 5 % of the KNN's, we prefer
# the polynomial (simpler and easier to interpret, see Unit 4b "Additional notes").
TOLERANCE = 0.05


class Batch:
    def __init__(self, batch_id, ph_opt, temperature_opt):
        self.batch_id = batch_id
        self.ph_opt = ph_opt
        self.temperature_opt = temperature_opt

        # Load the data
        self.df = pd.read_csv(filepath)
        self.X = self.df[FEATURES]
        self.y = self.df[TARGET]

        # Step 1 of the workflow: split into training and test sets (80 / 20)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

        # Filled in later
        self.searches = {}  # fitted GridSearchCV objects
        self.best_name = None  # name of the selected model
        self.final_model = None  # model used for predictions

    def display_info(self):
        print(f"Batch ID: {self.batch_id:4d} | "
              f"Optimal pH: {self.ph_opt:4.1f} | "
              f"Optimal Temperature: {self.temperature_opt:4.1f} C")

    # ------------------------------------------------------------------
    # Build a model: Pipeline + GridSearchCV
    # ------------------------------------------------------------------
    def make_search(self, name):
        """
        Returns a GridSearchCV object for the chosen model ("Polynomial" or "KNN").
        Both models are Pipelines that start with StandardScaler.
        GridSearchCV tries every hyperparameter value using 5-fold cross-validation.
        Scoring is RMSE so both models are compared with the same metric.
        """
        if name == "Polynomial":
            # Degree 1 is just a linear model, so Linear Regression is
            # already included in this search.
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('poly', PolynomialFeatures(include_bias=False)),
                ('model', LinearRegression())
            ])
            grid = {'poly__degree': [1, 2, 3, 4]}
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', KNeighborsRegressor())
            ])
            grid = {'model__n_neighbors': list(range(1, 21)),
                    'model__weights': ['uniform', 'distance']}

        return GridSearchCV(pipeline, grid, cv=5,
                            scoring='neg_root_mean_squared_error',
                            return_train_score=True)

    # ------------------------------------------------------------------
    # Phase 1: train, evaluate and select a model
    # ------------------------------------------------------------------
    def evaluate_and_select_model(self):
        """
        1. Fits GridSearchCV for each model on the TRAINING data only.
        2. Calculates RMSE and R2 on the training set and the test set.
        3. Selects a model using the VALIDATION RMSE from cross-validation.
        4. Saves a results table and figures.
        """
        rows = []
        for name in MODEL_NAMES:
            search = self.make_search(name)
            search.fit(self.X_train, self.y_train)
            self.searches[name] = search

            # Warn if the best value is at the edge of the grid
            # (Unit 4b: then a wider range might be even better)
            for param, values in search.param_grid.items():
                best_value = search.best_params_[param]
                if isinstance(best_value, int) and best_value == max(values):
                    print(f"[Note] {name}: best {param} = {best_value} is the largest "
                          f"value searched. Try a wider range.")

            train_pred = search.predict(self.X_train)
            test_pred = search.predict(self.X_test)

            rows.append({
                'Model': name,
                'Best Hyperparameters': str(search.best_params_),
                'Train RMSE': sf.calculate_rmse(self.y_train, train_pred),
                'Validation RMSE': -search.best_score_,
                'Test RMSE': sf.calculate_rmse(self.y_test, test_pred),
                'Train R2': r2_score(self.y_train, train_pred),
                'Test R2': r2_score(self.y_test, test_pred),
            })

        results = pd.DataFrame(rows)

        print("\n--- Model Evaluation (GridSearchCV, 5-fold) ---")
        for _, r in results.iterrows():
            print(f"{r['Model']:11s} | {r['Best Hyperparameters']}")
            print(f"{'':11s} | RMSE  train: {r['Train RMSE']:.4f}  "
                  f"validation: {r['Validation RMSE']:.4f}  test: {r['Test RMSE']:.4f}  [d^-1]")
            print(f"{'':11s} | R2    train: {r['Train R2']:.4f}  test: {r['Test R2']:.4f}")

        # ----- Select the model -----
        # We use the validation RMSE (not the test set) to choose, so the
        # test set stays untouched and only evaluates the final model.
        val_poly = results.loc[results['Model'] == 'Polynomial', 'Validation RMSE'].iloc[0]
        val_knn = results.loc[results['Model'] == 'KNN', 'Validation RMSE'].iloc[0]

        if val_poly <= val_knn * (1 + TOLERANCE):
            self.best_name = "Polynomial"
            print("\nSelected: Polynomial (similar or better validation RMSE than KNN, "
                  "and easier to interpret).")
        else:
            self.best_name = "KNN"
            print("\nSelected: KNN (its validation RMSE is clearly lower than the polynomial's).")

        # ----- Table and figures (before the final refit) -----

        sf.save_table(results, "model_performance_metrics.csv")
        sf.plot_model_comparison(results, self.best_name)
        sf.plot_validation_curve(self.searches["Polynomial"])
        sf.plot_model_curves(self.searches, self.X_train, self.y_train,
                             self.X_test, self.y_test, FEATURES)

        # Final model: best hyperparameters, trained on ALL the data
        self.final_model = self.searches[self.best_name].best_estimator_
        self.final_model.fit(self.X, self.y)


        if self.best_name == "Polynomial":
            sf.print_polynomial_equation(self.final_model, FEATURES)

        return results


    # ------------------------------------------------------------------
    # Learning curve: does adding more samples help?
    # ------------------------------------------------------------------
    def make_learning_curve(self):
        """
        Trains each model (with GridSearchCV) on an increasing number of
        training samples and records the test RMSE each time.
        """
        n_total = len(self.X_train)
        sizes = [int(n_total * fraction) for fraction in [0.2, 0.4, 0.6, 0.8, 1.0]]

        test_rmse = {name: [] for name in MODEL_NAMES}
        for n in sizes:
            for name in MODEL_NAMES:
                search = self.make_search(name)
                search.fit(self.X_train.iloc[:n], self.y_train.iloc[:n])
                test_pred = search.predict(self.X_test)
                test_rmse[name].append(sf.calculate_rmse(self.y_test, test_pred))

        sf.plot_learning_curve(sizes, test_rmse)


    # ------------------------------------------------------------------
    # Phase 2: predict with the selected model
    # ------------------------------------------------------------------
    def predict_growth_rate(self, temperature, ph, glucose, lactate):
        """Predicts the specific growth rate [d^-1] for new culture conditions."""
        if self.final_model is None:
            self.evaluate_and_select_model()

        # Warn if the new conditions are outside the range of the training data
        inputs = [temperature, ph, glucose, lactate]
        for column, value in zip(FEATURES, inputs):
            low, high = self.df[column].min(), self.df[column].max()
            if value < low or value > high:
                print(f"[Warning] {column} = {value} is outside the data range "
                      f"[{low:.2f}, {high:.2f}]. The prediction is an extrapolation.")

        new_culture_data = pd.DataFrame([inputs], columns=FEATURES)
        predicted_growth_rate = self.final_model.predict(new_culture_data)[0]

        print("\n--- Cell Growth Rate Prediction ---")
        print(f"Model used: {self.best_name}")
        print(f"Culture Conditions -> Temp: {temperature} C | pH: {ph} | "
              f"Glucose: {glucose} mM | Lactate: {lactate} mM")
        print(f"Predicted Specific Growth Rate: {predicted_growth_rate:.4f} d^-1")

        return predicted_growth_rate
