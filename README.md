# house_price_prediction--ml
Machine Learning project to predict house prices using California Housing dataset (scikit-learn).

# House Price Prediction using Machine Learning (California Housing Dataset)

## 📌 Project Overview
This is a Machine Learning regression project that predicts house prices using the **California Housing dataset** available in **scikit-learn** (`fetch_california_housing`).  
The project includes **data loading, preprocessing, exploratory data analysis (EDA), model training, and evaluation**.

---

## 🎯 Objective
To build a machine learning model that can predict **median house value** based on different housing-related features such as income, house age, rooms, location, and population.

---

## 📊 Dataset Information
- Dataset Name: **California Housing Dataset**
- Source: `sklearn.datasets.fetch_california_housing`
- Target Variable: **MedHouseVal** (Median House Value)
- Type: Regression Dataset

✅ Dataset is loaded directly from scikit-learn, so no external dataset file is required.

---

## ⚙️ Project Workflow
1. Import required libraries  
2. Load dataset using sklearn  
3. Convert dataset into Pandas DataFrame  
4. Data preprocessing (missing values handling, feature scaling etc.)  
5. Exploratory Data Analysis (EDA) & visualization  
6. Train-test split  
7. Model training (regression models)  
8. Model evaluation using metrics  
9. Final prediction results  

---

## 🧠 Machine Learning Models Used
- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor

---

## 📈 Evaluation Metrics
The model performance is measured using:

- **MSE (Mean Squared Error)**
- **R² Score**

---

## 🛠️ Tech Stack / Tools
- Python
- Google Colab
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn

---

## 🚀 How to Run This Project
1. Open the notebook in **Google Colab** or **Jupyter Notebook**
2. Run all cells step-by-step

Notebook file:
- `House_Price_Prediction.ipynb`

---

## ✅ Results
The trained model predicts housing prices based on input features and is evaluated using RMSE and R² score to select the best model.

---

## 🔮 Future Improvements
- Hyperparameter tuning (GridSearchCV / RandomizedSearchCV)
- Feature engineering
- Use advanced models like **XGBoost / CatBoost**
- Deploy model using **Streamlit** or **Flask**

---

## 👨‍💻 Author
**Nikhil Kumar**
