# Mental Health Analysis and Classification using Social Media Data

## 🧠 Project Overview

This repository contains a comprehensive Jupyter Notebook (`mental_health.ipynb`) that explores the intricate relationship between social media usage, lifestyle factors, and mental health. The project performs extensive Exploratory Data Analysis (EDA) on a synthetic dataset and builds multiple Machine Learning models to classify an individual's mental state based on their digital and physical habits.

The core finding highlights the critical need for handling class imbalance in mental health datasets to achieve robust and reliable model performance.

## 💾 Dataset

The analysis is based on the `mental_health_social_media_dataset.csv` file, which contains 5000 entries of individuals' daily activities and self-reported mental state scores.

### Key Features:

| Column Name | Description | Example Data |
| :--- | :--- | :--- |
| `daily_screen_time_min` | Total time spent on screens (minutes). | 320 |
| `social_media_time_min` | Time spent specifically on social media platforms (minutes). | 160 |
| `negative_interactions_count` | Number of negative online interactions. | 1 |
| `positive_interactions_count` | Number of positive online interactions. | 2 |
| `sleep_hours` | Average hours of sleep. | 7.4 |
| `physical_activity_min` | Minutes of physical activity per day. | 28 |
| `anxiety_level` | Self-reported anxiety score (1-10). | 2 |
| `stress_level` | Self-reported stress score (1-10). | 7 |
| `mood_level` | Self-reported mood score (1-10). | 6 |
| `platform` | Primary social media platform used (e.g., Instagram, Snapchat). | Instagram |
| `mental_state` (Target) | Categorical mental health state (e.g., `Stressed`, `Calm`, `Depressed`). | Stressed |

## 🛠️ Notebook Contents & Analysis Steps

The `mental_health.ipynb` notebook is structured into three main phases:

### 1. Data Loading & Preparation
* Imports necessary libraries (`pandas`, `matplotlib`, `seaborn`, `sklearn`).
* Loads and inspects the dataset for data types and missing values.
* Creates a new categorical feature, `age_group`, to facilitate demographic analysis.

### 2. Exploratory Data Analysis (EDA)
* Visualizes the distribution of categorical variables like **Gender**, **Platform**, **Age Group**, and the target variable **Mental State**.
* Analyzes patterns and correlations between key features (e.g., screen time, sleep hours, physical activity) and various mental health metrics (anxiety, stress, mood, etc.).

### 3. Machine Learning Classification

The project builds a classification model to predict the `mental_state` of an individual.

#### 🎯 Modeling Steps:
1.  **Preprocessing:** Categorical features (`gender`, `platform`, `mental_state`) are encoded using `LabelEncoder` for model training.
2.  **Model Training:** The data is split, and **five different classifiers** are trained on the original, **unbalanced dataset**:
    * Logistic Regression
    * Decision Tree
    * Random Forest
    * Support Vector Machine (SVM)
    * K-Nearest Neighbors (KNN)
3.  **Handling Imbalance:** **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to the training data to address the significant class imbalance in the `mental_state` target variable.
4.  **Re-Evaluation:** All five classifiers are re-trained and their performance is evaluated on the **SMOTE-balanced dataset**.
5.  **Comparison:** Confusion matrices are generated to visually compare the performance of each model before and after SMOTE.

#### ⭐ Key Finding

The notebook demonstrates that models trained on the **SMOTE-balanced dataset** show **significantly improved performance** metrics, highlighting the critical role of handling class imbalance for accurate mental health predictions.
