# HTL Bio-Oil Yield Prediction using Machine Learning

### A Machine Learning Framework for Predicting Bio-Oil Yield from Hydrothermal Liquefaction Experiments

---

**Author**

Rohan

Department of Chemical Engineering

Indian Institute of Technology Kharagpur

---

**Project Duration**

June – July 2026

---

**Technologies Used**

Python • Scikit-Learn • XGBoost • CatBoost • SHAP • Pandas • NumPy • Matplotlib

---

**Keywords**

Hydrothermal Liquefaction, Bio-Oil Yield Prediction, Machine Learning, Regression, Explainable AI, XGBoost, Random Forest

---

# Abstract

Hydrothermal Liquefaction (HTL) is a promising thermochemical conversion process for transforming wet biomass into bio-oil, offering a sustainable pathway for renewable fuel production. However, accurately predicting bio-oil yield remains challenging due to the complex interactions between biomass composition, operating conditions, and process parameters. Experimental optimization is often expensive and time-consuming, motivating the use of data-driven predictive models.

This project presents a modular machine learning framework for predicting bio-oil yield using a publicly available Hydrothermal Liquefaction dataset containing 2,284 experimental observations and 29 input features describing feedstock properties and processing conditions. Multiple regression algorithms, including Linear Regression, Gradient Boosting, Random Forest, Extra Trees, XGBoost, and CatBoost, were implemented and systematically evaluated. Hyperparameter optimization using Randomized Search Cross Validation was performed to improve the performance of the tree-based models.

To improve interpretability, explainable artificial intelligence (XAI) techniques including SHAP values, Partial Dependence Plots (PDP), Feature Importance analysis, and Permutation Importance were employed to identify the most influential variables governing bio-oil yield. Lipid content, reaction temperature, Higher Heating Value (HHV), protein content, and fatty acid concentration were identified as the dominant predictors.

Among all evaluated models, the tuned XGBoost regressor achieved the highest predictive performance with a Test R² score of 0.8689 and a Test Mean Absolute Error (MAE) of 4.2589. The resulting framework demonstrates that ensemble tree-based learning methods can accurately model the nonlinear relationships inherent in HTL processes while providing interpretable insights into the governing process variables.

The developed framework is modular, reproducible, and extensible, making it suitable for future studies involving catalyst optimization, process parameter optimization, and biomass feedstock screening.

---

# 1. Introduction

The increasing demand for sustainable energy sources and growing concerns regarding climate change have accelerated research into renewable alternatives to fossil fuels. Among the various renewable energy technologies, biomass has emerged as a promising resource because of its abundance, carbon neutrality, and ability to be converted into a wide range of valuable fuels and chemicals. Efficient utilization of biomass can significantly contribute to reducing greenhouse gas emissions while improving global energy security.

Hydrothermal Liquefaction (HTL) is an advanced thermochemical conversion process that transforms wet biomass into an energy-dense liquid fuel known as bio-oil. Unlike conventional pyrolysis processes, HTL operates under high temperature and pressure in the presence of water, eliminating the need for energy-intensive drying of biomass feedstocks. This makes HTL particularly attractive for processing high-moisture materials such as algae, sewage sludge, food waste, and agricultural residues.

Despite its advantages, the HTL process involves complex interactions among feedstock composition, reaction temperature, residence time, heating profile, solvent properties, and several other operating conditions. These nonlinear interactions make accurate prediction of bio-oil yield difficult using traditional empirical equations or first-principles models. Experimental optimization of HTL conditions is also expensive, labor-intensive, and time-consuming, as each experiment requires specialized equipment and careful control of reaction parameters.

Recent advances in machine learning have provided powerful tools for modeling complex nonlinear systems directly from experimental data. Ensemble learning algorithms such as Random Forest, Gradient Boosting, Extra Trees, and XGBoost have demonstrated excellent predictive capabilities across numerous engineering applications by learning intricate relationships without requiring explicit mathematical models. In addition to achieving high predictive accuracy, modern explainable artificial intelligence (XAI) techniques allow researchers to interpret model decisions and identify the process variables that most strongly influence system performance.

This project applies machine learning techniques to predict bio-oil yield from a comprehensive Hydrothermal Liquefaction experimental dataset containing 2,284 experiments and 29 input features describing biomass composition and operating conditions. Multiple regression algorithms are implemented within a modular machine learning framework, systematically evaluated using consistent preprocessing and performance metrics, and further improved through hyperparameter optimization.

Beyond prediction accuracy, the project emphasizes interpretability by incorporating SHAP analysis, Partial Dependence Plots (PDP), Feature Importance analysis, and Permutation Importance. These explainability techniques provide valuable insight into the relative influence of biomass composition and processing conditions on bio-oil production, enabling both accurate prediction and scientific understanding of the HTL process.

The resulting framework demonstrates how modern machine learning methods can accelerate process optimization, reduce experimental effort, and support data-driven decision-making in biomass conversion research. The modular architecture developed in this work also provides a reusable foundation for future studies involving catalyst optimization, feedstock screening, process optimization, and broader applications of machine learning in sustainable energy systems.

## 1.1 Objectives

The primary objectives of this project are:

- Develop a modular machine learning framework for predicting bio-oil yield from HTL experiments.
- Compare the predictive performance of multiple regression algorithms under a common evaluation pipeline.
- Improve model performance through systematic hyperparameter optimization.
- Interpret model predictions using explainable AI techniques such as SHAP, Partial Dependence Plots, Feature Importance, and Permutation Importance.
- Identify the most influential process variables governing bio-oil yield.
- Build a reusable and extensible codebase that can support future HTL and biomass research.

# 2. Literature Review

Hydrothermal Liquefaction (HTL) has become an important research area due to its ability to convert wet biomass directly into liquid biofuels without requiring energy-intensive drying. Over the past two decades, researchers have investigated the influence of biomass composition, reaction temperature, residence time, catalysts, and solvent conditions on bio-oil production. These studies have established HTL as one of the most promising technologies for sustainable biofuel generation.

Early HTL research primarily focused on experimental investigations aimed at understanding reaction mechanisms and optimizing operating conditions. Numerous studies demonstrated that lipid-rich biomass generally produces higher bio-oil yields than carbohydrate-rich feedstocks, while reaction temperature significantly affects both product yield and product quality. However, these experimental approaches require considerable time, specialized equipment, and financial resources, limiting the number of operating conditions that can be practically explored.

To overcome these limitations, researchers have increasingly adopted data-driven methods for modeling HTL processes. Machine learning algorithms are particularly attractive because they can learn complex nonlinear relationships directly from experimental observations without requiring explicit physical models. Compared with conventional statistical regression techniques, modern ensemble learning methods provide significantly improved predictive accuracy while remaining computationally efficient.

Several machine learning algorithms have been applied to biomass conversion problems, including Random Forest, Gradient Boosting, Support Vector Regression, Artificial Neural Networks, and XGBoost. Ensemble tree-based methods consistently demonstrate strong performance because they effectively capture nonlinear interactions among biomass composition, reaction conditions, and process variables while exhibiting good robustness to noisy experimental data.

In recent years, explainable artificial intelligence (XAI) has become increasingly important in scientific machine learning. Although highly accurate predictive models are valuable, researchers also seek to understand why predictions are made and which variables contribute most strongly to model behavior. Techniques such as SHAP (SHapley Additive exPlanations), Partial Dependence Plots (PDP), Feature Importance analysis, and Permutation Importance have become widely adopted for interpreting complex machine learning models. These approaches provide quantitative insight into variable importance and improve confidence in data-driven decision-making.

Despite significant progress, many published studies focus primarily on achieving high predictive accuracy while providing limited discussion of model interpretability, reproducibility, or software organization. Furthermore, several implementations are developed as standalone notebooks, making them difficult to extend or reproduce for future research.

The present work addresses these limitations by developing a modular and reusable machine learning framework that integrates data preprocessing, model training, hyperparameter optimization, comparative evaluation, and explainability within a unified pipeline. Multiple regression algorithms are evaluated under identical preprocessing and evaluation protocols, while interpretability techniques are employed to identify the process variables that most strongly influence bio-oil yield. This combination of predictive performance, reproducibility, and explainability distinguishes the proposed framework from many existing implementations.

## 2.1 Research Gap

From the existing literature, several research gaps can be identified:

- Many studies focus primarily on maximizing predictive accuracy while providing limited model interpretability.
- Hyperparameter optimization is often absent or performed only for a single machine learning algorithm.
- Comparative evaluation across multiple regression models using a consistent preprocessing pipeline is relatively limited.
- Many published implementations are developed as exploratory notebooks rather than modular and reusable software frameworks.
- Explainable AI techniques are not consistently integrated into the complete machine learning workflow.

This project addresses these gaps by combining multiple regression algorithms, systematic hyperparameter optimization, comprehensive model comparison, and explainable AI within a modular software architecture that is both reproducible and extensible.

# 3. Dataset Description

## 3.1 Dataset Source

The dataset used in this study is a publicly available Hydrothermal Liquefaction (HTL) experimental database compiled by the Pacific Northwest National Laboratory (PNNL). The database contains experimentally measured HTL results collected from numerous published research studies and represents one of the most comprehensive datasets available for bio-oil yield prediction.

The dataset consists of 2,284 experimentally validated HTL observations collected under different feedstock compositions and operating conditions. Each observation corresponds to an individual HTL experiment with associated biomass characteristics, reaction parameters, and measured product yields.

The primary objective of this work is to predict the bio-oil yield (%) produced during the HTL process.

---

## 3.2 Dataset Characteristics

The final dataset used for model development contains:

| Property | Value |
|-----------|------:|
| Number of Samples | 2284 |
| Number of Input Features | 29 |
| Target Variable | Bio-Oil Yield (%) |
| Machine Learning Task | Regression |

The dataset includes experiments performed using various biomass feedstocks such as microalgae, macroalgae, wood, food waste, sewage sludge, herbaceous biomass, and mixed biomass resources.

---

## 3.3 Feature Categories

The input variables can be broadly classified into the following categories.

### Biomass Composition

These variables describe the chemical composition of the biomass feedstock.

- Carbohydrates
- Cellulose
- Hemicellulose
- Lignin
- Proteins
- Lipids
- Ash
- Sugar
- Amino Acids
- Fatty Acids
- Guaiacol
- Glycerol
- Carboxylic Acids

### Feedstock Information

These variables describe the origin and characteristics of the biomass.

- Resource
- Details
- Origin
- Method
- Higher Heating Value (HHV)

### Operating Conditions

These variables represent the HTL reaction conditions.

- Temperature
- Heating Rate
- Heating Time
- Holding Time
- Total Reaction Time
- Heating Profile
- Dry Matter
- Solvent
- Solvent Polarity
- Additive
- Additive Quantity

---

## 3.4 Target Variable

The response variable used for prediction is:

**Bio-Oil Yield (%)**

This continuous variable represents the percentage yield of bio-oil obtained after completion of the HTL process.

The prediction of bio-oil yield is important because it directly reflects the efficiency of biomass conversion into renewable liquid fuel.

---

## 3.5 Data Preprocessing

Several preprocessing steps were performed before model training.

### Missing Value Handling

- Numerical variables were imputed using the median value.
- Categorical variables were imputed using the most frequent category.

### Encoding

Categorical variables were transformed using One-Hot Encoding to enable their use in machine learning algorithms.

### Data Splitting

The dataset was divided into:

- Training Set: 70%
- Testing Set: 30%

using a fixed random seed to ensure reproducibility.

### Machine Learning Pipeline

The preprocessing operations and regression model were combined into a single Scikit-Learn Pipeline. This ensured that identical preprocessing transformations were consistently applied during both training and prediction, eliminating the possibility of data leakage.

---

## 3.6 Dataset Summary

The dataset contains substantial diversity in both biomass composition and operating conditions. Such diversity enables machine learning algorithms to learn complex nonlinear relationships between feedstock characteristics, reaction parameters, and bio-oil yield.

The relatively large number of experimental observations, combined with multiple chemical composition descriptors and process variables, makes this dataset well suited for developing robust regression models capable of generalizing across different HTL operating conditions.

# 4. Exploratory Data Analysis

Exploratory Data Analysis (EDA) was performed to understand the statistical characteristics of the dataset, identify relationships between variables, detect potential outliers, and guide the subsequent machine learning workflow. Multiple visualization techniques were employed to investigate the distribution of the target variable, relationships among numerical features, biomass composition, and variations across different feedstock categories.

---

## 4.1 Distribution of Bio-Oil Yield

The distribution of the bio-oil yield was examined to understand the range and variability of the prediction target.

The analysis indicated that bio-oil yield spans a broad range of values, reflecting the considerable diversity of HTL operating conditions and biomass feedstocks represented in the dataset. Most observations are concentrated within the moderate yield range, while comparatively fewer experiments produce extremely low or extremely high yields.

The absence of severe skewness suggests that the target variable is suitable for regression modelling without requiring logarithmic or power transformations.

**Figure:** Distribution of Bio-Oil Yield

**Figure 4.1:** Distribution of bio-oil yield.

![Oil Distribution](figures/oil_distribution.png)

---

## 4.2 Correlation Analysis

A Pearson correlation heatmap was generated to investigate relationships among the numerical variables.

The analysis revealed that several biomass composition variables exhibit moderate correlations due to their inherent chemical relationships. However, no pair of variables demonstrated excessive correlation that would justify feature removal solely to reduce multicollinearity.

The correlation analysis also indicated that bio-oil yield depends on the combined influence of multiple variables rather than any single dominant parameter, motivating the use of nonlinear machine learning algorithms.

**Figure:** Correlation Heatmap

**Figure 4.2:** Correlation heatmap of numerical features.

![Correlation Heatmap](figures/correlation_heatmap.png)

---

## 4.3 Biomass Resource Analysis

The dataset contains experiments performed on multiple biomass feedstocks, including microalgae, macroalgae, wood, agricultural residues, sewage sludge, food waste, and mixed biomass resources.

Resource-wise analysis revealed noticeable differences in average bio-oil yield across biomass categories. Feedstocks rich in lipid content generally exhibited higher oil yields, whereas lignocellulosic feedstocks displayed comparatively lower yields.

These observations agree with established HTL literature and provide confidence regarding the quality of the dataset.

**Figure:** Bio-Oil Yield across Biomass Resources

**Figure 4.3:** Bio-oil yield across different biomass resources.

![Resource Analysis](figures/resource_vs_oil.png)

---

## 4.4 Biomass Composition Analysis

The chemical composition of biomass strongly influences HTL performance. Therefore, detailed analysis was performed on major compositional variables including:

- Lipids
- Proteins
- Carbohydrates
- Cellulose
- Hemicellulose
- Lignin
- Ash

The analysis revealed substantial variation among feedstocks, indicating that composition descriptors contain valuable predictive information for machine learning models.

The generated ternary composition plots further illustrated the diversity of biomass samples and highlighted the wide coverage of chemical compositions represented in the dataset.

**Figures:**

- Biomass Composition Analysis
- Ternary Composition Plot

**Figure 4.4:** Biomass composition ternary plot.

![Ternary Plot](figures/ternary_composition.png)

**Figure 4.5:** Biomass composition distribution.

![Composition Histogram](figures/composition_sum_histogram.png)

---

## 4.5 Temperature and Process Variables

Scatter plots were generated to examine relationships between process parameters and bio-oil yield.

Reaction temperature exhibited a noticeable nonlinear relationship with bio-oil yield. Similarly, variables such as heating time, holding time, and total reaction time demonstrated complex interactions that could not be adequately described using simple linear models.

These observations provided strong motivation for selecting ensemble tree-based regression algorithms capable of learning nonlinear decision boundaries.

**Figures:**

- Oil Yield vs Temperature
- Oil Yield vs HHV
- Oil Yield vs Lipids

**Figure 4.6:** Relationship between reaction temperature and bio-oil yield.

![Oil vs Temperature](figures/oil_vs_temperature.png)

**Figure 4.7:** Relationship between lipid content and bio-oil yield.

![Oil vs Lipids](figures/oil_vs_lipids.png)

**Figure 4.8:** Relationship between Higher Heating Value and bio-oil yield.

![Oil vs HHV](figures/oil_vs_hhv.png)

---

## 4.6 Distribution Comparison

Violin plots were used to compare the distributions of major process outputs and biomass characteristics.

Unlike standard box plots, violin plots simultaneously display summary statistics and probability density, providing deeper insight into data variability and the presence of multimodal distributions.

The analysis demonstrated considerable variability across experimental conditions while revealing relatively few extreme outliers that would require removal.

**Figures:**

- Violin Plot – Bio-Oil Yield
- Violin Plot – Char Yield

**Figure 4.9:** Violin plot of bio-oil yield.

![Violin Oil](figures/violin_oil.png)

**Figure 4.10:** Violin plot of char yield.

![Violin Char](figures/violin_char.png)

---

## 4.7 Composition Consistency

A broken-axis visualization was employed to investigate biomass composition percentages.

The analysis confirmed that most biomass samples exhibit chemically consistent compositions, while only a small number of observations deviated substantially from expected ranges. These outliers were documented separately for transparency but retained because they correspond to experimentally reported values.

This analysis ensured that potentially informative observations were not removed unnecessarily during preprocessing.

**Figure:** Broken-Axis Composition Analysis

**Figure 4.11:** Broken-axis visualization of biomass composition.

![Broken Axis](figures/composition_broken_axis.png)

---

## 4.8 Summary of EDA

The exploratory analysis provided several important insights that guided the remainder of the study:

- Bio-oil yield exhibits substantial variability across experiments.
- Biomass composition plays a critical role in determining oil yield.
- Reaction temperature is one of the most influential operating variables.
- Relationships between process variables and oil yield are predominantly nonlinear.
- The dataset contains diverse biomass feedstocks, improving model generalization.
- No severe multicollinearity or data quality issues were identified that required major feature removal.

These observations strongly motivated the use of ensemble tree-based machine learning algorithms and explainable artificial intelligence techniques in the subsequent modelling phase.

# 5. Methodology

## 5.1 Overview

The objective of this project was to develop a modular and reproducible machine learning framework capable of accurately predicting bio-oil yield from Hydrothermal Liquefaction (HTL) experiments.

Unlike conventional notebook-based implementations, the proposed framework separates data loading, preprocessing, model training, evaluation, hyperparameter optimization, visualization, and explainability into independent reusable modules. This modular design improves code maintainability, reproducibility, and extensibility while allowing new machine learning algorithms to be incorporated with minimal modifications.

The overall workflow adopted in this project is illustrated below.

```

Raw Dataset

↓

Data Cleaning & Preprocessing

↓

Feature Engineering

↓

Train-Test Split

↓

Model Training

↓

Hyperparameter Optimization

↓

Model Evaluation

↓

Explainability Analysis

↓

Model Comparison Dashboard

↓

Final Results

5.2 Software Architecture

The project follows a modular software architecture consisting of four primary components.

Core Module

The core package provides reusable utilities that form the foundation of the machine learning framework.

Its responsibilities include:

Dataset loading
Data preprocessing
Pipeline construction
Performance metrics
Model factory
Input/Output utilities
Experiment logging
Generic model trainer
Hyperparameter search framework

This design ensures that common functionality is implemented only once and reused across all machine learning models.

Models Module

The models package contains implementations of all regression algorithms evaluated in this study.

Implemented models include:

Linear Regression
Gradient Boosting Regressor
Random Forest Regressor
Extra Trees Regressor
XGBoost Regressor
CatBoost Regressor

Each model shares the same preprocessing pipeline and evaluation methodology, ensuring a fair comparison of predictive performance.

Visualization Module

The visualization package generates publication-quality figures used for exploratory analysis and model interpretation.

Implemented visualizations include:

Correlation Heatmap
Feature Importance
SHAP Analysis
Partial Dependence Plots
Permutation Importance
Error Distribution
Parity Plot
Violin Plots
Model Comparison Dashboard
Project Summary Dashboard
Experiments Module

The experiments package manages hyperparameter optimization and comparative evaluation.

Its responsibilities include:

Randomized Search Cross Validation
Performance comparison
Result logging
Saving optimized models
Cross-validation summaries
5.3 Data Preprocessing Pipeline

Before model training, several preprocessing operations were applied to ensure data consistency.

Missing Value Imputation

Numerical variables were imputed using the median value, while categorical variables were imputed using the most frequent category.

This approach preserves all experimental observations without discarding incomplete records.

Categorical Encoding

Categorical variables were transformed using One-Hot Encoding.

This encoding enables tree-based regression algorithms to effectively utilize categorical process information while avoiding arbitrary ordinal relationships.

Pipeline Construction

All preprocessing operations were integrated into a Scikit-Learn Pipeline.

Using a pipeline provides several advantages:

Eliminates data leakage
Ensures identical preprocessing during training and prediction
Improves reproducibility
Simplifies model deployment
5.4 Machine Learning Models

Multiple regression algorithms were evaluated to identify the most suitable predictive model for HTL bio-oil yield prediction.

Linear Regression

Served as the baseline model for comparison.

Gradient Boosting

Introduced sequential ensemble learning capable of capturing nonlinear relationships.

Random Forest

Used bootstrap aggregation and randomized feature selection to reduce variance and improve predictive performance.

Extra Trees

Introduced additional randomness during tree construction to increase diversity among ensemble members.

XGBoost

Implemented gradient boosting with advanced regularization techniques to improve predictive accuracy and reduce overfitting.

CatBoost

Evaluated gradient boosting with native categorical feature handling to investigate its suitability for HTL data.

5.5 Hyperparameter Optimization

Model performance was further improved using Randomized Search Cross Validation.

Randomized Search was selected because it efficiently explores large hyperparameter spaces while requiring significantly fewer evaluations than exhaustive Grid Search.

The optimization procedure employed:

Repeated K-Fold Cross Validation
Randomized parameter sampling
Parallel computation
Performance evaluation using R² score

Hyperparameter tuning was performed for:

Random Forest
XGBoost

The optimized models demonstrated improved predictive performance compared with their default counterparts.

5.6 Model Evaluation

Regression models were evaluated using two complementary performance metrics.

Coefficient of Determination (R²)

R² measures the proportion of variance explained by the predictive model.

Higher values indicate better predictive capability.

Mean Absolute Error (MAE)

MAE measures the average magnitude of prediction errors.

Lower values indicate better prediction accuracy.

Evaluating both metrics provides a balanced assessment of model performance.

5.7 Explainable Artificial Intelligence

To improve model transparency, several explainability techniques were incorporated into the workflow.

SHAP Analysis

Quantified the contribution of individual features to each prediction while providing global feature importance.

Partial Dependence Plots

Illustrated how changes in individual variables influence predicted bio-oil yield.

Permutation Importance

Measured the reduction in predictive performance caused by randomly permuting each feature.

Feature Importance

Tree-based feature importance provided an additional global ranking of influential variables.

Together, these techniques improve confidence in the predictive models while providing scientific insight into HTL process behaviour.

5.8 Reproducibility

Several design decisions were incorporated to ensure experimental reproducibility.

Fixed random seed throughout the workflow.
Unified preprocessing pipeline.
Modular project architecture.
Automatic experiment logging.
Automatic model serialization.
Automatic generation of figures and prediction files.

These practices enable the complete workflow to be reproduced with minimal user intervention while facilitating future extensions of the framework.

# 6. Experimental Results and Discussion

## 6.1 Experimental Setup

All machine learning models were trained and evaluated using the same preprocessing pipeline, train-test split, and evaluation metrics to ensure a fair comparison. Numerical features were imputed using the median value, categorical variables were one-hot encoded, and the resulting preprocessing pipeline was integrated with each regression model using the Scikit-Learn Pipeline framework.

Model performance was evaluated using the coefficient of determination (R²) and Mean Absolute Error (MAE). Hyperparameter optimization was performed using Randomized Search Cross Validation for the Random Forest and XGBoost models.

---

## 6.2 Model Performance Comparison

Table 6.1 summarizes the performance of all evaluated regression models.

| Model | Train R² | Test R² | Train MAE | Test MAE |
|------|---------:|---------:|----------:|----------:|
| Random Forest | 0.9812 | 0.8624 | 1.6252 | 4.4111 |
| Tuned Random Forest | 0.9959 | 0.8659 | 0.6685 | 4.2796 |
| XGBoost | 0.9803 | 0.8634 | 1.8881 | 4.4094 |
| **Tuned XGBoost** | **0.9864** | **0.8689** | **1.4909** | **4.2589** |
| Extra Trees | 0.9987 | 0.8619 | 0.1334 | 4.1621 |
| CatBoost | 0.9238 | 0.8370 | 3.7792 | 5.1441 |

**Table 6.1:** Performance comparison of all evaluated machine learning models.

The comparison demonstrates that all ensemble tree-based methods substantially outperform traditional linear regression approaches, confirming the highly nonlinear relationship between biomass composition, operating conditions, and bio-oil yield.

Among all evaluated models, the tuned XGBoost regressor achieved the highest Test R² score of **0.8689**, indicating that it explained approximately 87% of the variability in bio-oil yield on unseen experimental data.

**Figure 6.1:** Overall Project Summary Dashboard

![Project Summary](figures/project_summary_dashboard.png)

**Figure 6.2:** Comparison of Test R² across all evaluated machine learning models.

![Model Comparison R2](figures/model_r2_comparison.png)

**Figure 6.3:** Comparison of Test MAE across all evaluated machine learning models.

![Model Comparison MAE](figures/model_mae_comparison.png)

---

## 6.3 Effect of Hyperparameter Optimization

Hyperparameter optimization produced measurable improvements for both Random Forest and XGBoost.

For Random Forest:

- Test R² improved from **0.8624** to **0.8659**
- Test MAE decreased from **4.4111** to **4.2796**

For XGBoost:

- Test R² improved from **0.8634** to **0.8689**
- Test MAE decreased from **4.4094** to **4.2589**

Although the numerical improvements appear modest, they demonstrate that systematic optimization of tree depth, learning rate, sampling strategies, and ensemble size improves model generalization while maintaining robustness.

**Figure 6.4:** Parity plot comparing actual and predicted bio-oil yield values.

![Parity Plot](figures/parity_plot.png)

**Figure 6.5:** Error distribution of prediction residuals.

![Error Distribution](figures/error_distribution.png)

---

## 6.4 Model Comparison

The experimental results reveal several important observations.

### Tuned XGBoost

The tuned XGBoost model achieved the highest predictive performance among all evaluated algorithms.

Its gradient boosting framework effectively captured the nonlinear interactions between biomass composition and operating conditions while maintaining good generalization on unseen data.

---

### Tuned Random Forest

The tuned Random Forest achieved performance very close to the tuned XGBoost model.

Its ensemble of decision trees provides high predictive accuracy while remaining relatively simple to interpret, making it an attractive alternative when computational efficiency and model simplicity are important.

---

### Extra Trees

The Extra Trees model achieved the lowest Test MAE among all evaluated models.

However, its nearly perfect training performance (Train R² ≈ 0.999) compared with a lower Test R² indicates mild overfitting.

Although Extra Trees accurately fits the training data, its ability to generalize to previously unseen experiments is slightly lower than the tuned XGBoost model.

---

### CatBoost

CatBoost produced the lowest predictive performance among the evaluated ensemble methods.

One possible explanation is that after one-hot encoding, the dataset contains relatively few high-cardinality categorical variables, reducing CatBoost's natural advantage over other gradient boosting algorithms.

---

## 6.5 Explainability Analysis

Model interpretability was investigated using multiple explainable AI techniques.

SHAP analysis, Partial Dependence Plots, Feature Importance, and Permutation Importance consistently identified similar variables as the dominant contributors to bio-oil yield prediction.

The five most influential variables identified through permutation importance were:

1. Lipids
2. Temperature
3. Higher Heating Value (HHV)
4. Proteins
5. Fatty Acids

The consistency among multiple explainability techniques increases confidence that these variables genuinely influence HTL bio-oil production rather than representing artifacts of a particular machine learning algorithm.

**Figure 6.6:** SHAP summary plot showing global feature importance.

![SHAP Beeswarm](figures/shap_beeswarm.png)

**Figure 6.7:** SHAP feature importance ranked by average absolute SHAP value.

![SHAP Bar](figures/shap_bar.png)

**Figure 6.8:** Permutation importance ranking of input variables.

![Permutation Importance](figures/permutation_importance.png)

**Figure 6.9:** Partial Dependence Plot for Lipids.

![PDP Lipids](figures/pdp_Lipids.png)

**Figure 6.10:** Partial Dependence Plot for Temperature.

![PDP Temperature](figures/pdp_Temperature.png)

**Figure 6.11:** Partial Dependence Plot for Proteins.

![PDP Proteins](figures/pdp_Proteins.png)

**Figure 6.12:** Partial Dependence Plot for Carbohydrates.

![PDP Carbohydrates](figures/pdp_CarboHydrates.png)

**Figure 6.13:** Partial Dependence Plot for Higher Heating Value (HHV).

![PDP HHV](figures/pdp_HHVResource.png)

---

## 6.6 Discussion

The results demonstrate that ensemble learning algorithms are highly effective for predicting HTL bio-oil yield.

Unlike linear regression models, tree-based ensemble methods successfully capture the complex nonlinear interactions among biomass composition, reaction temperature, heating conditions, and feedstock characteristics.

The explainability analysis further confirms that biomass composition plays a dominant role in determining bio-oil yield, with lipid-rich feedstocks consistently producing higher predicted yields. Reaction temperature also emerged as one of the most influential operating parameters, highlighting the importance of process optimization during HTL.

The modular software framework developed in this project ensures that additional regression algorithms, optimization strategies, or explainability methods can be incorporated with minimal modifications. Consequently, the framework serves not only as a predictive tool but also as a reusable research platform for future investigations into hydrothermal liquefaction and biomass conversion.

Overall, the experimental results demonstrate that the combination of modern ensemble learning methods and explainable artificial intelligence provides an accurate, interpretable, and reproducible approach for predicting bio-oil yield from HTL experiments.

# 7. Conclusion

This project presented a comprehensive machine learning framework for predicting bio-oil yield from Hydrothermal Liquefaction (HTL) experiments. A publicly available experimental dataset containing 2,284 observations and 29 input features was used to develop and evaluate multiple regression models under a unified preprocessing and evaluation pipeline.

Several machine learning algorithms, including Random Forest, Extra Trees, XGBoost, CatBoost, Gradient Boosting, and Linear Regression, were systematically compared. Hyperparameter optimization was performed using Randomized Search Cross Validation for the Random Forest and XGBoost models, resulting in measurable improvements in predictive performance.

Among all evaluated models, the tuned XGBoost regressor achieved the highest predictive accuracy with a Test R² of **0.8689** and a Test MAE of **4.2589**, demonstrating its ability to effectively capture the nonlinear relationships between biomass composition, operating conditions, and bio-oil yield. The tuned Random Forest model also produced highly competitive results, while Extra Trees achieved the lowest prediction error but exhibited mild overfitting.

Beyond predictive accuracy, this work emphasized model interpretability through the integration of Explainable Artificial Intelligence (XAI) techniques. SHAP analysis, Partial Dependence Plots, Feature Importance, and Permutation Importance consistently identified lipid content, reaction temperature, Higher Heating Value (HHV), protein content, and fatty acid concentration as the most influential variables governing bio-oil production.

A major contribution of this work is the development of a modular and reusable machine learning framework. Unlike notebook-based implementations, the framework separates data preprocessing, model training, hyperparameter optimization, visualization, and explainability into independent components, improving maintainability, reproducibility, and extensibility.

Overall, the results demonstrate that ensemble tree-based machine learning algorithms, combined with modern explainability techniques, provide an accurate and interpretable approach for bio-oil yield prediction. The developed framework can serve as a foundation for future HTL research and broader applications of machine learning in biomass conversion and sustainable energy systems.

# 8. Future Work

Several opportunities exist for extending the present work.

- Evaluate additional machine learning models such as LightGBM, TabNet, and deep neural networks.
- Investigate Bayesian Optimization as an alternative to Randomized Search for hyperparameter tuning.
- Incorporate catalyst composition and catalyst concentration as additional predictive features where available.
- Extend the framework to simultaneously predict multiple HTL products, including bio-oil, char, gas, and aqueous phase yields using multi-output regression.
- Explore uncertainty estimation techniques to quantify prediction confidence.
- Develop an interactive web application using Streamlit for real-time bio-oil yield prediction.
- Integrate the framework with optimization algorithms to recommend operating conditions that maximize bio-oil yield.
- Expand the dataset with newly published HTL experiments to improve model robustness and generalization.

These extensions would further enhance both the predictive capability and practical applicability of the proposed framework.

# References

1. Peterson, A. A., Vogel, F., Lachance, R. P., Fröling, M., Antal Jr., M. J., & Tester, J. W. (2008). Thermochemical biofuel production in hydrothermal media.

2. Elliott, D. C. Hydrothermal liquefaction of biomass.

3. Breiman, L. (2001). Random Forests.

4. Friedman, J. H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.

5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.

6. Prokhorenkova, L., Gusev, G., Vorobev, A., Dorogush, A. V., & Gulin, A. (2018). CatBoost: Unbiased Boosting with Categorical Features.

7. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions.

8. Pedregosa, F., et al. (2011). Scikit-Learn: Machine Learning in Python.

# Appendix

## A. Project Directory Structure

```
HTL_BioOil_Prediction/
├── data/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── *.csv
├── report/
├── src/
│   ├── core/
│   ├── experiments/
│   ├── legacy/
│   ├── models/
│   └── visualization/
├── README.md
├── requirements.txt
└── .gitignore
```

## B. Generated Outputs

The framework automatically generates:

- Trained model files
- Prediction CSV files
- Experiment summary tables
- SHAP visualizations
- Partial Dependence Plots
- Permutation Importance plots
- Feature Importance plots
- Model comparison dashboard
- Project summary dashboard

## C. Software Environment

- Python 3.11
- Scikit-Learn
- XGBoost
- CatBoost
- SHAP
- Pandas
- NumPy
- Matplotlib