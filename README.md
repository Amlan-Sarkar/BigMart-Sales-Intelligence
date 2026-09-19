# BigMart-Sales-Intelligence-Prediction

A machine learning project that predicts **product sales across BigMart outlets** using the BigMart Sales dataset from Kaggle.

The project covers data preprocessing, exploratory analysis, feature engineering, machine learning, model evaluation, and deployment through a Streamlit application.

## Project Overview

Retail businesses need accurate sales estimates to better understand product and outlet performance.

This project uses historical BigMart sales data to build a regression model that predicts `Item_Outlet_Sales` based on product and outlet characteristics such as:

* Product category and type
* Item MRP
* Item visibility
* Item weight
* Fat content
* Outlet type
* Outlet size
* Outlet location
* Outlet age

The final model is evaluated on **unseen products** using group-based validation to provide a more realistic estimate of predictive performance.

## Dataset

**Source:** [BigMart Sales Data — Kaggle](https://www.kaggle.com/datasets/brijbhushannanda1979/bigmart-sales-data)

The dataset contains sales information for products across BigMart outlets.

* **Training records:** 8,523
* **Test records:** 5,681
* **Unique products:** 1,559
* **Outlets:** 10
* **Target:** `Item_Outlet_Sales`

## Technologies Used

* **Python**
* **Pandas** — data manipulation
* **NumPy** — numerical computing
* **Matplotlib & Seaborn** — data visualization
* **Scikit-learn** — preprocessing, model training and evaluation
* **XGBoost** — gradient boosting regression
* **Joblib** — model persistence
* **Streamlit** — interactive prediction application
* **Jupyter Notebook** — analysis and experimentation

## Project Workflow

```text
Data Collection
      ↓
Data Cleaning & Preprocessing
      ↓
Exploratory Data Analysis
      ↓
Feature Engineering
      ↓
Train Regression Models
      ↓
Group-Based Evaluation
      ↓
Model Comparison
      ↓
Save Best Model
      ↓
Streamlit Prediction App
```

## Data Preprocessing

The dataset was cleaned and prepared before model training.

Key preprocessing steps include:

* Imputing missing `Item_Weight` values using product-level averages
* Filling remaining missing weights with the dataset median
* Handling missing `Outlet_Size` values
* Standardizing inconsistent `Item_Fat_Content` labels
* Creating `Outlet_Age` from the establishment year
* Encoding categorical variables for machine learning

## Model Evaluation

Several regression models were evaluated:

| Model             |         MAE |          RMSE |         R² |
| ----------------- | ----------: | ------------: | ---------: |
| **Random Forest** | **₹768.96** | **₹1,110.93** | **0.5880** |
| XGBoost           |     ₹777.28 |     ₹1,118.75 |     0.5822 |
| Linear Regression |     ₹917.85 |     ₹1,233.22 |     0.4923 |
| Mean Baseline     |   ₹1,375.82 |     ₹1,731.40 |    -0.0007 |

The **Random Forest Regressor** achieved an R² score of **0.5880** on the held-out set.

### Validation Strategy

Because the dataset contains multiple records for the same products across different outlets, a standard random split can allow the same product to appear in both training and validation data.

To reduce this leakage, the project uses:

* `GroupShuffleSplit` for the train/test split
* `GroupKFold` for cross-validation
* `Item_Identifier` as the grouping variable

This evaluates how the model performs when predicting sales for **products that were not present during training**.

### Cross-Validation

5-fold GroupKFold results:

* **R²:** 0.5859 ± 0.0125
* **MAE:** ₹766.01 ± ₹9.25

## Feature Importance

The Random Forest model identified the following features as the most influential:

| Feature        | Importance |
| -------------- | ---------: |
| Item_MRP       |      51.1% |
| Outlet_Type    |      31.9% |
| Outlet_Age     |       7.2% |
| Other features |      ~9.8% |

`Item_MRP` and `Outlet_Type` were the two most influential features in the model.

## Streamlit Application

The project includes a Streamlit application that loads the trained model and preprocessing artifacts to generate sales predictions.

Run the application with:

```bash
streamlit run app.py
```

## Project Structure

```text
BigMart-Sales-Intelligence/
│
├── data/
│   ├── train.csv
│   └── test.csv
│
├── notebook/
│   └── Amlan Sarkar_BigMart Sales Intelligence Prediction.ipynb
│
├── models/
│   ├── rf_model.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   ├── metrics.pkl
│   ├── test_results.csv
│   └── feature_importance.csv
│
├── app.py
├── README.md
└── requirements.txt
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd BigMart-Sales-Intelligence
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Run the Notebook

```bash
jupyter notebook
```

Open:

```text
notebook/Amlan Sarkar_BigMart Sales Intelligence Prediction.ipynb
```

## Run the Streamlit App

```bash
streamlit run app.py
```

## Key Takeaways

* Built an end-to-end retail sales prediction pipeline.
* Performed data cleaning and feature engineering on real-world retail data.
* Compared multiple regression algorithms.
* Used group-based validation to evaluate performance on unseen products.
* Achieved an R² score of **0.5880** with Random Forest.
* Created a Streamlit application for model-based predictions.

## Limitations

* The dataset represents BigMart sales from 2013.
* The model does not incorporate current market conditions or temporal sales trends.
* Sales depend on additional factors that are not available in the dataset.
* Predictions should therefore be interpreted within the context of the original dataset.

## License

This project is intended for **educational and portfolio purposes**.

Dataset: BigMart Sales Data, available through Kaggle.
