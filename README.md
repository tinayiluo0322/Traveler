# Traveler Insurance Modeling Competition

## Overview

This repository contains the final models developed for the Traveler Insurance Modeling Competition. The models can be found in the `/finalmodels` directory.

## Background

The goal of the competition was to create a rating plan for InsNova Auto Insurance Company, utilizing historical auto claim data.

## Objective

The primary objective was to provide a reliable method for predicting the claim cost associated with each policy.

## Dataset

### InsNova Data Set

The dataset used in the competition is based on one-year vehicle insurance policies spanning from 2004 to 2005. The dataset includes:

- Total policies: 45,239
- Policies with at least one claim: Approximately 6.8%

## Repository Structure
- Each team member tried out different models to approach the objective and work done can be found in respective folders `<first_name>TravelerInsurnace`.

## Models Tried

### Sampling Techniques in Random Forest

- **Oversampling & Under-sampling:** Methods to balance the dataset.
- **Two-Part Implementation:** With and without separate parts for model training.

### Decision Tree

Traditional decision tree models for classification or regression.

### Zero-Inflated Models

Specifically for discrete outcomes, not continuous. This includes:

- Poisson Regression
- Negative Binomial Regression
- Zero-Inflated Normal Regression

### Tobit Regression

A censored regression model, suitable for limited dependent variables.

### Hurdle Regression

A two-component model combining:

- Logistic Regression for non-zero cost occurrence.
- Gamma Regression for positive responses (cost amount among non-zero observations).

### Tweedie Regression

- In R: Emphasizing distribution assumption, model assumption, and influential points.
- In Python: Utilizing random search and grid search methods.

### CatBoost Models

- Standard CatBoost.
- CatBoost with Grid Search: For hyperparameter tuning.
- CatBoost with Tweedie Grid Search: Integrating Tweedie distribution.

### Frequency * Severity CatBoost Model

- Frequency calculated using a Poisson distribution.
- Severity calculated using a Gamma distribution.

### LightGBM

A gradient boosting framework that uses tree-based learning algorithms.

### XGBoost

Another efficient and scalable implementation of gradient boosting.

## Final Models

### Tweedie with Random Search

Utilizing Python for a Tweedie distribution model with random search optimization.

### CatBoost with Grid Search

Implementing CatBoost algorithm with grid search for hyperparameter tuning in Python.

### CatBoost with Tweedie Grid Search

Combining CatBoost with Tweedie distribution, using grid search for optimization in Python.


