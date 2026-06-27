
# Supervised Learning End-to-End Playbook

<br />

## About this guide

The purpose of this document is to offer a beginner-friendly, end-to-end playbook for supervised learning projects — a structured, replicable set of steps with practical rules of thumb to guide decisions along the way.

In this guide, you'll find multiple options with the most common techniques, practical rules of thumb, common pitfalls, and some code snippets (mostly pandas/scikit-learn).

A few things to keep in mind:
- Training a Machine Learning model is not a linear process (it is an iterative process). In this document we try to offer a simple step-by-step guide, but sometimes you may need to iterate back and forth.
- The guide includes some quick rules of thumb with values that can be used as a reference. Use them as guidelines rather than strict limits.
- Every project is different. You may need to adapt to your specific case, apply your own criteria, and do things in a different way.

<br />
Ready to dive in? Let's go! 🚀


<br /><br /><br />


## 1. Define the Problem 🎯

- What is the target variable?
- Is it Classification (e.g., spam detection) or Regression (e.g., house prices) ?
- What metric defines success? (e.g., R2, RMSE, F1-score, ROC-AUC...)


<br /><br />



## 2. Collect Data 🗂️

- Gather relevant data (ensure size and quality)


<br /><br />



## 3. Exploratory Data Analysis (EDA) 🔍

- Start with a quick overview. For example:
    - `df.head()`, `df.tail()`, and/or `df.sample()` → inspect example rows from the dataset
    - `df.shape` → see the number of rows and columns
    - `df.columns` → list all columns
- Identify the target variable
- Check if there's class imbalance (classification problems only)
    - `df['my_target'].value_counts(normalize=True) * 100` → get the percentage distribution of each class in the column "my_target".
- Check data types and structure
    - `df.dtypes` → get a list of column data types (numeric, object, etc.)
    - `df.info()` → get a list of columns, data types, and how many non-null values each column has
- Check for missing values
    - `df.isna().sum()` → get the number of missing values per column
- Explore data distributions and basic statistics
    - `df.describe()` → summary stats for numeric columns (mean, std, min, max, quartiles). Check if there's outliers or anything suspicious (weird min/max, unexpected means, etc). For example, a person aged 200, a negative price, or 150% discount can signal dirty data.
    - `df.nunique()` → number of unique values per column (helps detect categorical vs ID-like features)
    - `df['column_name'].value_counts()` → shows category frequencies; useful for detecting dominant classes, rare labels, or messy/inconsistent values.
    - Visualize data using plots (histograms, bar plots, scatter plots, box plots) → understand distributions, relationships, and outliers
- Identify correlations between features (e.g., using a correlation heatmap) → detect multicollinearity (mostly for numeric features)
- Check if you have enough data for Machine Learning
    - Basic rule of thumb:
        - Aim for at least 10–20 samples per feature for traditional ML models.
        - Example: 50 features → ideally 500–1,000 training samples at the very least
        - This is a minimum guideline, not a guarantee of good performance. 
        - Many problems require much more data, especially when the task is complex, the model is complex, or data quality is low.
    - You can find further guidelines on data requirements [here](https://claude.ai/share/0b710685-ad38-4cd9-bbb9-9d2e173dddce)
    - If the data does not seem enough, consider gathering more data or training a simple baseline model to check whether performance is acceptable.
    <!-- 
    Further notes on data requirements:
    - https://github.com/luisjunco/DS-ML-Course-Ironhack/blob/main/b%20-%20thins-learned/ML-supervised-learning/0b%20-%20ML%20Data%20Requirement%20Guidelines.ipynb
    -->
- Finally, summarize insights. Either write down a few key takeaways or at least do a quick mental recap with the most important things. For example:
    - What's the target variable
    - What features look important
    - Do you need to handle class imbalance
    - Do you need to convert data to the correct data type
    - Do you need to handle missing values and/or outliers


⚠️ If you spot anything that needs fixing (class imbalance, wrong data types, missing values, outliers…), don't address it yet. In this step, the goal is to understand the data and identify potential issues — not to fix them. We'll do that in the next steps.

<br /><br />



## 4. Data Splitting ✂️

<!--
@todo: 
- explain cross-validation better (and/or address it in the section Hyperparameter tuning)
- can also include some guidelines / rule of thumb, e.g.:
    - if you want a very simple approach → train/test split
    - small/medium datasets (e.g. < 10k samples) → train/test split + cross-validation (in hyperparameter tuning)
    - large datasets → train/validation/test split is usually enough
-->

Common patterns to split data:
- a) **Train/Test split**: 
    - Simple and common for quick experiments, but limited when you also want to compare models or tune hyperparameters carefully.
    - Example: 80% of the data used for training + 20% for testing
- b) **Train/Validation/Test split**: 
    - A very practical workflow when you want one set for training, one for model selection, and one final untouched set for evaluation.
    - Example: 70% of the data used for training + 15% for validation/hyperparameter tuning + 15% reserved for the final testing
- c) **Cross-validation**: 
    - Often gives a more robust estimate of performance, especially on smaller datasets, but it also adds more complexity.
    - Example: 80% of the data used for training, and later reused for hyperparameter tuning with k-fold cross-validation + 20% reserved for the final testing

⚠️ Important: Always keep the test set untouched until the end and only use it for final evaluation (to ensure an unbiased performance estimate).

<!--
Note: In practice, pipelines (available in scikit-learn and most ML frameworks) are strongly recommended —especially with cross-validation— because they ensure all preprocessing steps (e.g., scaling, imputation, encoding) are applied only on the training data within each fold. This prevents data leakage and guarantees a consistent and correct evaluation process.
-->

<br><br>

Special cases:
- Classification problems:
    - The split should preserve the proportion of each class across train, validation, and test sets — this is called stratification. For example, if only 5% of your data is labeled "fraud", a random split might leave your test set with very few fraud cases. 
    - In scikit-learn, you can do this by passing stratify=y to train_test_split.
- Time series data:
    - Do not use a random split. The split should respect time order, so the model is always trained on the past and evaluated on the future.
    - An essential rule: "Never let the future leak into the past — random splitting for time series is a form of data leakage"
- Grouped data:
    - If multiple rows belong to the same person, customer, device, or company, those groups usually should not be split across train, validation, and test sets.
- Very small datasets:
    - Cross-validation is often preferred because a single validation split may waste too much data or give unstable results.

<br /><br />



## 5. Data Cleaning 🧹 

<!--
note:
- for simplicity, consider moving "Data Cleaning" before "Data Splitting",
and mention in which cases it needs to be done after splitting (in particular, mention that if you do imputation, it needs to be done AFTER data splitting)
-->

- Remove obvious irrelevant columns
    - Remove features that are clearly irrelevant (e.g., IDs, names, constant features...)
- Ensure Correct Data Types
    - Inspect data types, and convert to the correct data type if needed.
    - For example, if a numeric column is stored as string, convert to int/float; if a date is stored as object convert to datetime, etc.
- Handle Missing Values. Common options to handle missing data:
    - a) Removal (dropping rows or columns). Typically:
        - Drop columns with high missing rates (e.g., >30–50%), especially if the column isn't crucial, or if there's high correlation with other features.
        - Then handle rows with missing values (drop or impute):
            - If you're confident the data will be available at prediction time (i.e., some rows are missing info now, but you'll have complete data when your model is used to make predictions with real-world data) → dropping those rows may be the easiest option. Note: after removing rows with missing data, check that it will not cause major data loss or bias (e.g., for classification, check that dropping rows doesn't create class imbalance).
            - If you think the data may not be available at prediction time → use imputation instead, since dropping isn't an option in production.
            - If in doubt → use imputation (explained below)
    - b) Imputation (replacing missing values with estimated ones). General rules of thumb:
        - If few missing values & categorical: use the mode
        - If few missing values & numerical & normal distribution: use the mean
        - If few missing values & numerical & skewed: use the median
        - If many missing values, consider other methods (e.g. KNN Imputation)
        - ⚠️ To prevent data leakage, any information used for imputation must be learned only from the training set (for example, if you need to compute the mean, use only the training data). Then use those same learned values to fill missing values in both the training set and the test set.
        - More info and techniques: https://dataaspirant.com/data-imputation-techniques/
    - c) Treat missing as its own category:
        - Instead of imputing, encode missing values as a separate category (e.g., “Unknown” or “Not recorded”). This is useful when the absence of a value may itself be informative.
        - Example: if "income" is missing in a credit risk model, it may reflect a specific customer behavior (e.g., reluctance to disclose income) that correlates with default risk. By keeping "Missing" as its own category, the model can learn patterns associated with that absence instead of masking them through imputation.
    - Note: After this step, you usually want no missing values in your dataset. Most standard models (like linear/logistic regression, KNN, and neural networks) do not natively handle missing data and require explicit preprocessing. A notable exception is tree-based gradient boosting models — XGBoost, LightGBM, and CatBoost — which have built-in mechanisms to handle missing values directly.

- Evaluate Outliers: Decide whether to keep, transform, or remove them (only remove if they are clearly errors, like a person's age being 200).


<br /><br />



## 6. Establish a Baseline Model 📏


Before you implement more complex algorithms, it is a good practice to establish a baseline model. A baseline model is a simple, quick-to-build model used as a reference point for performance.


Why we define a baseline model:
- Helps detect data issues early (e.g., if the baseline performs unexpectedly well or poorly)
- Sets a minimum performance level (it’s a "don’t be worse than this" reference point for all later models)
- Helps measure real improvement: you can quantify gains from feature engineering or better algorithms
- Fast to build: useful early in a project before heavy work is done

If a more complex model does not clearly outperform this baseline, it may indicate that the data has limited predictive signal, that a simple model is sufficient for this particular problem, or that the more complex model has not been properly tuned or trained.


Some common options to establish a baseline model:
- a) Using a "dummy" baseline
    - Typically: predicting the mean (regression) or the majority class (classification)
    - Advantage: very simple to implement; sets a minimum performance bar
- b) Using a simple Linear Model
    - Typically: Linear Regression (regression), or Logistic Regression (classification)
    - Advantage: fast, interpretable, and often a solid baseline
    - Note: Linear models often require numerical input and are sensitive to feature scale → encode categorical features + scale numerical features before training (both steps are covered later in this cheatsheet, but if you choose a linear model as a baseline, you'll need to do them here — and potentially again after feature engineering)
- c) Using a Tree-Based Model
    - Typically: Decision Tree, or Random Forest
    - Advantage: solid baseline without requiring feature scaling
    - Note: tree-based models do not require scaling but many implementations require numerical input → encode categorical features before training (encoding is covered later in this cheatsheet, but if you choose a tree-based model as a baseline, you'll need to do it here — and potentially again after feature engineering)

These options are compatible. In practice, many teams start with a dummy baseline alongside a linear or tree-based model.


Once built, the baseline model should be evaluated using the same metric(s) defined at the start of the project. This score becomes the reference point that all subsequent models must beat.


<br /><br />


## 7. Feature Engineering 🏗️

- Encoding / Handling Categorical Data (most ML algorithms require numerical input, so categories are usually converted into numbers). These are the most common options:
    - a) Drop the feature:
        - If a categorical feature has little predictive value or too many unique values that are hard to encode, removing that feature may be a valid option.
    - b) One-Hot Encoding:
        - Creates a separate binary column for each category, where only the column matching the category is set to 1 and all others are 0
        - Example: color = red → [red=1, blue=0, green=0]
        - A common variant is **dummy encoding**, which drops one column per feature (the reference category). For example, with 3 colors, instead of 3 columns you get 2 — red is implied when blue=0 and green=0. This avoids perfect multicollinearity between the encoded columns, which can matter for linear models such as logistic regression.
    - c) Label Encoding:
        - Assigns an integer to each category
        - Example: red=0, blue=1, green=2
        - It's simple and efficient, but it can mislead models that interpret numbers as having magnitude or order (e.g., linear models, distance-based models...) because it turns unordered categories into numbers that the model may wrongly interpret as having meaningful order or distance.
    - d) Ordinal Encoding: 
        - Assigns each category an integer based on a meaningful order. 
        - Example: small = 0, medium = 1, large = 2
        - It is used when the categories have a natural ranking, so the numeric values correctly represent relative order, even if the distance between them is not necessarily equal.
    - e) Target Encoding: 
        - Replaces each category with a value calculated from the target variable for that category (commonly the mean). 
        - It is useful for categorical features with many unique values and can capture the relationship between the category and the target. However, it can easily overfit if not done carefully (typically requires cross-validation and/or smoothing).
    <!--
    - Some notes for categorical data: https://github.com/luisjunco/DS-ML-Course-Ironhack/blob/main/b%20-%20thins-learned/ML-supervised-learning/2c%20-%20Feature%20Engineering%20-%20How%20to%20handle%20categorical%20data.ipynb
    -->

- Signal Amplification (transform or combine existing features to make relevant patterns in the data more pronounced and easier for a model to learn). Some common techniques used for Signal Amplification:
    - a) Non-linear transformations:
        - Transform features mathematically to capture non-linear relationships or reduce skew, making patterns easier for models to learn
        - For example, if we're creating a model to predict energy consumption, and we suspect the relationship with temperature is not linear (very cold and very hot days both increase usage), we can add a polynomial feature to capture this curved effect: `df["temperature_squared"] = df["temperature"] ** 2`
    - b) Interaction terms:
        - Combine two or more features to capture joint effects that are not visible in individual features alone
        - For example, if we're creating a model to predict the risk of heart disease, and we know that the risk of heart disease is higher not just from high blood pressure or high cholesterol alone, but especially when both are high at the same time, we can create a new feature that reflects if a person has both of them at the same time: `df["blood_pressure_cholesterol_interaction"] = df["blood_pressure"] * df["cholesterol"]`
    - c) Binning:
        - Turning continuous numerical values into categories
        - Useful when there are clear thresholds or meaningful ranges in the data (e.g., grouping age into "child / adult / senior"), or when you want simpler, more interpretable features.
        - Avoid when the exact numeric value carries important signal or when models already capture non-linearities effectively (e.g., tree-based models)
    - Tips: 
        - If you know something about the phenomena that you're studying, try to find the way of encoding it in the data. Signal Amplification can really make the difference between a decent and a great model ⭐️
        - Domain knowledge is often key. In real-world projects it's common to consult domain experts (e.g. doctors, finance analysts, product teams) to design meaningful features like ratios or transformations that reflect how the system actually behaves.

- Feature Scaling (adjust numerical features so they share a similar range or scale). Common techniques:
    - a) Normalization (`MinMaxScaler`): 
        - Rescales values to a fixed range, usually between 0 and 1.
        - Normalization is useful for distance-based models or neural networks, but it’s sensitive to outliers. 
    - b) Standardization (`StandardScaler`):
        - Transforms values so they have a mean of 0 and a standard deviation of 1. This centers the data and puts all features on the same scale, making it easier for algorithms to compare them fairly — regardless of their original units or ranges.
        - Standardization is often preferred as a default option because it's more robust to outliers than normalization and works well with most algorithms.
    - ⚠️ To prevent data leakage, feature scaling should be fit only on the training data, and then applied to both train and test sets. In scikit-learn, use `fit_transform()` on the training set and `transform()` on validation and test sets.


<br /><br />



## 8. Feature Selection 🧲

- Avoid multicollinearity:
    - Perform a correlation analysis to detect highly correlated feature pairs. (e.g. plot a correlation heatmap). 
    - If two features convey nearly identical information, drop one of them.
    - ⚠️ To prevent data leakage, use only the train data for your correlation analysis.

- Remove low-signal features:
    - Drop features that are unlikely to help the model make good predictions.
    - One sign is very low variance (if a feature is almost constant across all rows, it may carry little very information).
    - Another sign is when a feature seems to have little or no clear relationship with the target. Note: even if a feature seems to have a low correlation with the target variable, it may still be useful when combined with other features or for certain value ranges (some algorithms like tree-based models catch these patterns better than others). Before dropping a feature, try to apply domain knowledge and think about whether it makes sense in the real world.

- (Optional) Dimensionality Reduction:
    - Dimensionality Reduction allows you to combine or transform features into a smaller set while preserving the most important information. The most common technique is Principal Component Analysis (PCA).
    - When to use it: If you have many features (e.g., 50+) and suspect the model is overfitting or running slowly
    - ⚠️ Always fit dimensionality reduction on the training data only, then apply it to the test set. Never fit on the combined dataset — this causes data leakage.
    - Trade-off: Dimensionality reduction makes your model faster and can reduce overfitting, but the resulting features become harder to interpret (you lose the ability to explain what each feature means)
    - Recommendation: Start without it. Only use if your baseline model clearly overfits or your dataset has too many features to manage.


<br /><br />



## 9. Handle Class Imbalance (Classification problems only) ⚖️

In classification, if one class is much more frequent than others (e.g., 95% "not fraud" vs. 5% "fraud"), we may need to take action to ensure the model performs well.


### 9.1 Check class distribution

First, start by checking the distribution of your data:
- `df['my_target'].value_counts(normalize=True) * 100` → get the percentage distribution of each class in the column "my_target".

Rule of thumb:
- 40% – 50% (Balanced): No action required. Standard models usually perform well.
- 20% – 40% (Mild imbalance): Usually fine. Most algorithms handle this without intervention.
- 5% – 20% (Moderate imbalance): You should consider handling imbalance.
- < 5% (Severe imbalance): Strong action is usually required (most models will struggle otherwise).


### 9.2. Check baseline model performance

To understand if you need to handle class imbalance, you should also evaluate your baseline model using the right metrics:
- In imbalanced datasets, accuracy can be misleading. 
- Instead, use other metrics like the confusion matrix, precision/recall, F1-score, ROC-AUC or PR-AUC.

Key sign that you should handle class imbalance:
- If your baseline model performs well on overall accuracy but fails to correctly identify the minority class. This is usually visible through low recall for the minority class and a confusion matrix showing few correct minority predictions (i.e., many false negatives or strong bias toward the majority class).


### 9.3. Decide if you need to handle class imbalance 

Use the findings from the previous steps to decide if you need to handle class imbalance. 

Some general guidelines:
- Balanced dataset + strong minority-class metrics (e.g., high recall/F1) → no action needed
- Imbalanced dataset + weak minority-class performance (e.g., low recall, many false negatives) → action required
- Imbalanced dataset + strong minority-class metrics → action is usually not required, but it may yield marginal gains


### 9.4. Some options to handle class imbalance 

If you need to address class imbalance, you can apply data-level methods or algorithm-level methods.

Data-Level Methods (resampling the data):
- a) Undersample the majority class (discard rows) — simple, but loses data
- b) Oversample the minority class (duplicate rows) — simple, but risks overfitting
- c) SMOTE (generate synthetic minority samples) — often a better option than undersampling or oversampling
- Notes:
    - If you decide to resample, it can be done now (before you start training and comparing models). Another option is to train and compare models first, and resample only if the performance is not as good as expected.
    - ⚠️ If you resample, make sure to do it only on the training data. The test set must remain untouched and representative of the true real-world distribution.

Algorithm-Level Methods (adjusting the model):
- a) Class weights — penalize mistakes on the minority class more (e.g., in scikit-learn many models have an option `class_weight='balanced'` to balance class weights)
b) Threshold tuning — define the minority class as the positive class (the class for which the model outputs a probability and applies the decision cutoff) and adjust the probability threshold (e.g., from 0.5 to 0.3). Lowering the threshold increases minority class predictions (higher recall), while raising it makes the model more conservative.
- c) Combine "Class weights" + "Threshold tuning" — a solid option for severe imbalance.
- Notes:
    - If you decide to apply algorithm-level methods, the moment to do it will be when you train models and/or during Hyperparameter Tuning.

Some general guidelines:
- If the dataset is balanced and/or minority-class metrics are strong → you may not need to do anything
- If you need to handle class imbalance → start with class weights
- If minority recall is still poor → try threshold tuning
- If still not good → try SMOTE

⚠️ Remember that, in imbalanced datasets, accuracy is not a reliable metric. Make sure to evaluate your model and validate improvements using appropriate metrics (e.g. confusion matrix, precision/recall, F1-score, ROC-AUC, PR-AUC...).


<br /><br />



## 10. Train and Compare Different Models 🤖

<!-- todo: explain briefly how to check underfitting/overfitting + some rules of thumb -->

- Try different algorithms (e.g. Linear/Logistic Regression, KNN, Decision Tree, Random Forest, XGBoost...). For each of them, train the model and calculate its corresponding metric/s (remember to check for underfitting/overfitting)
- Then, compare the metric obtained with the baseline model to see if it's actually a better option.

⚠️ For model evaluation, do not use the test set (yet) — the test set should only be used in the final evaluation to get a reliable performance estimate

Notes:
- Consider integrating **regularization** techniques (e.g., L1/L2 penalties) into your model choice and training strategy to prevent overfitting, especially for complex models or when working with limited or noisy data.
- A common approach is to train and compare different algorithms using their default settings. Once we've identified the best-performing models, we will apply hyperparameter tuning only to those selected models — this saves time and avoids unnecessary work, since weaker models usually won't outperform stronger ones even after tuning.


<br /><br />



## 11. Hyperparameter Tuning 🔧

Next, apply hyperparameter tuning to the models that performed best in the previous step.

Hyperparameter tuning is the process of trying different settings to get the best performance from a specific algorithm (e.g., tree depth in decision trees, number of neighbors in KNN...).

To compare and choose hyperparameters, it's a good practice to use a validation set or cross-validation. The choice depends on how you did data splitting:
- If you used a Train/Test split (e.g. 80/20): → use the training set + cross-validation
- If you used a Train/Validation/Test split (e.g. 70/15/15): → use the validation set


Some options for hyperparameter tuning:
- Manual Search (manually try specific values based on intuition or domain knowledge; useful for quick checks or when the number of combinations is very small)
- Grid Search (tries all combinations within a predefined grid of values)
- Random Search (samples random combinations)


⚠️ When evaluating and comparing different hyperparameter settings, do not use the test set (reserve the test set for the final evaluation only).


<br /><br />



## 12. Final Evaluation on Test Set 🏆

- Once satisfied with tuning, evaluate the model's ultimate performance on the completely unseen test set. This provides an unbiased estimate of how well the model will generalize to new data.


<br /><br />



## 13. Model Deployment 🚀

- Export the trained model and integrate it into an application (API, web app, etc.).
- Remember to apply the same data preprocessing to new inputs as was used during training (e.g., scaling, encoding), so that the model receives data in the correct format and produces reliable predictions.


<br /><br />



## 14. Monitor & Maintain 🔄

- Track model performance over time to detect degradation (data drift, concept drift).
- If needed, retrain or update the model periodically with new data to keep it accurate.



<br /><hr /><hr /><br />



## Resources

- ML Project Checklist for Production Success (medium):
    - https://medium.com/@nirvana.elahi/ml-project-checklist-for-production-success-bde509a2f33e

- Machine Learning Lifecycle (geeksforgeeks):
    - https://www.geeksforgeeks.org/machine-learning/machine-learning-lifecycle/

- Scikit-Learn Cheat Sheet (datacamp):
    - https://www.datacamp.com/cheat-sheet/scikit-learn-cheat-sheet-python-machine-learning
    - includes code snippets for preprocessing data, split data, create the model, fit the model, etc


