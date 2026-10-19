# =======================================================================
# FUNCTIONS
# Define reusable helper functions in this file.
# =======================================================================
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

TABLES_DIR = "tables"
FIGURES_DIR = "figures"


# ----------------------------------------------------------------------
# Simple helpers
# ----------------------------------------------------------------------
def calculate_average(limits):
    return sum(limits) / len(limits)


def calculate_rmse(y_true, y_pred):
    """Root mean square error (same units as the target)."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def save_table(table, filename):
    """Saves a DataFrame as a CSV file in the 'tables' folder."""
    os.makedirs(TABLES_DIR, exist_ok=True)
    path = os.path.join(TABLES_DIR, filename)
    table.to_csv(path, index=False)
    print(f"[Success] Saved table to '{path}'")


def save_figure(filename):
    """Saves the current matplotlib figure in the 'figures' folder."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Success] Saved figure to '{path}'")


# ----------------------------------------------------------------------
# Figure 1: compare the models (bias-variance)
# ----------------------------------------------------------------------
def plot_model_comparison(results, best_name):
    """
    Grouped bar chart of the RMSE on the training set, validation folds and test set.
    - High train and high test RMSE      -> underfitting
    - Low train RMSE but high test RMSE  -> overfitting
    """
    models = results['Model'].tolist()
    x = np.arange(len(models))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, results['Train RMSE'], width, label='Train', edgecolor='black')
    plt.bar(x, results['Validation RMSE'], width, label='Validation (CV)', edgecolor='black')
    plt.bar(x + width, results['Test RMSE'], width, label='Test', edgecolor='black')

    plt.xticks(x, models)
    plt.ylabel("RMSE [d$^{-1}$] (lower is better)")
    plt.title(f"Model Comparison (selected: {best_name})", fontweight='bold')
    plt.legend()
    save_figure("model_comparison.png")


# ----------------------------------------------------------------------
# Figure 2: validation curve for the polynomial degree
# ----------------------------------------------------------------------
def plot_validation_curve(search):
    """
    Plots training and validation RMSE for each polynomial degree tried by GridSearchCV.
    The scores are stored as negative RMSE, so we flip the sign.
    """
    degrees = search.cv_results_['param_poly__degree'].data.astype(int)
    train_rmse = -search.cv_results_['mean_train_score']
    validation_rmse = -search.cv_results_['mean_test_score']

    plt.figure(figsize=(8, 5))
    plt.plot(degrees, train_rmse, 'o-', label='Training')
    plt.plot(degrees, validation_rmse, 's-', label='Validation')
    plt.xticks(degrees)
    plt.xlabel("Polynomial degree")
    plt.ylabel("RMSE [d$^{-1}$]")
    plt.title("Validation Curve: Polynomial Degree", fontweight='bold')
    plt.legend()
    save_figure("validation_curve.png")


# ----------------------------------------------------------------------
# Figure 3: learning curve
# ----------------------------------------------------------------------
def plot_learning_curve(sizes, test_rmse):
    """
    Test RMSE vs number of training samples for each model.
    If the curve is still dropping, more samples would likely help.
    """
    plt.figure(figsize=(8, 5))
    for name, values in test_rmse.items():
        plt.plot(sizes, values, 'o-', label=name)
    plt.xlabel("Number of training samples")
    plt.ylabel("Test RMSE [d$^{-1}$]")
    plt.title("Learning Curve", fontweight='bold')
    plt.legend()
    save_figure("learning_curve.png")


# ----------------------------------------------------------------------
# Figure 4: effect of each feature (works for any model, including KNN)
# ----------------------------------------------------------------------

def plot_model_curves(searches, X_train, y_train, X_test, y_test, feature_names):
    """
    One panel per feature. Each panel shows the training samples, the test samples
    and both models, varying that feature while the others are held at their mean.
    """
    means = X_train.mean()
    styles = {'Polynomial': ':', 'KNN': '--'}

    plt.figure(figsize=(11, 9))
    for i, feature in enumerate(feature_names):
        values = np.linspace(X_train[feature].min(), X_train[feature].max(), 100)

        # 100 copies of the mean sample, then change only one column
        data = pd.DataFrame([means.values] * 100, columns=feature_names)
        data[feature] = values

        plt.subplot(2, 2, i + 1)
        plt.scatter(X_train[feature], y_train, color='lightgray', label='Training samples')
        plt.scatter(X_test[feature], y_test, color='green', marker='D',
                    edgecolor='black', label='Test samples')
        for name, search in searches.items():
            plt.plot(values, search.predict(data), styles[name], label=f"Model ({name})")

        plt.xlabel(feature)
        plt.ylabel("Specific growth rate [d$^{-1}$]")
        if i == 0:
            plt.legend(fontsize=8)

    plt.suptitle("Model Predictions vs Samples (other features held at mean)",
                 fontweight='bold')
    save_figure("model_curves.png")
# ----------------------------------------------------------------------
# Interpreting the polynomial model (Unit 4b, "Interpreting linear models")
# ----------------------------------------------------------------------
def print_polynomial_equation(pipeline, feature_names):
    """
    Prints the polynomial weights, largest first. Because the features were
    standardized, the weights can be compared with each other.
    """
    term_names = pipeline.named_steps['poly'].get_feature_names_out(feature_names)
    weights = pipeline.named_steps['model'].coef_
    intercept = pipeline.named_steps['model'].intercept_

    print("\n--- Polynomial Model Weights (standardized features) ---")
    print(f"Baseline (intercept): {intercept:+.4f}")

    order = np.argsort(np.abs(weights))[::-1]
    for i in order[:10]:
        print(f"{weights[i]:+.4f} * {term_names[i]}")
    if len(weights) > 10:
        print(f"... ({len(weights) - 10} smaller terms not shown)")
