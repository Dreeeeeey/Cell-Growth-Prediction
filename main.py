# =====================================================================
# MAIN PROGRAM
# =====================================================================

import src.functions as sf  # This allows you to access functions from functions.py using sf.function_name()

from src.classes import Batch  # This allows you to import individual Classes from classes.py

from src.constants import PH_LIMITS, TEMPERATURE_LIMITS  # This imports all constants from constants.py

ph_optimal = sf.calculate_average(PH_LIMITS)
temperature_optimal = sf.calculate_average(TEMPERATURE_LIMITS)

batch = Batch(
    batch_id=1,
    ph_opt=ph_optimal,
    temperature_opt=temperature_optimal
)
batch.display_info()

# Phase 1: train Polynomial and KNN models with GridSearchCV, evaluate them,
# select one, and save the table and figures
model_results = batch.evaluate_and_select_model()

# Learning curve: would more training samples improve the models?
batch.make_learning_curve()

# Phase 2: predict the growth rate with the selected model.
# Glucose and lactate are in mM, matching the dataset columns.
batch.predict_growth_rate(
    temperature=temperature_optimal,
    ph=ph_optimal,
    glucose=13,  # mM
    lactate=2.1  # mM
)