# 🎓 College Graduation Rate Prediction Using Ridge and LASSO Regression

![College Graduation Rate Prediction](https://github.com/SYEDFAIZAN1987/College-Graduation-Rate-Prediction/blob/main/pic%201.png)

## 📘 About the Project

This project employs **Ridge** and **LASSO Regression** to predict college graduation rates based on a rich dataset of U.S. colleges. The project uses the **College dataset** from the **ISLR** package, which contains information about U.S. colleges and universities, including tuition fees, student demographics, and graduation rates.

By utilizing regularization techniques, this project addresses issues of multicollinearity, selects significant predictors, and provides actionable insights for improving educational outcomes.

---

## 🔑 Key Features

1. **Data Preprocessing**:
   - Comprehensive exploration and cleaning of the dataset.
   - Analysis of multicollinearity and feature engineering.

2. **Modeling Techniques**:
   - Ridge Regression (L2 Regularization).
   - LASSO Regression (L1 Regularization).
   - Stepwise Regression for model comparison.

3. **Model Evaluation**:
   - Comparison of RMSE for Ridge, LASSO, and Stepwise Regression.
   - Validation of overfitting and generalization using training and testing datasets.

4. **Visualization**:
   - Detailed plots showing regularization paths, deviance explained, and variable importance.

---

## 📊 Key Insights

- **Best Model**: Ridge Regression outperformed other models with the lowest RMSE.
- **Significant Predictors**:
  - **Private**: Private colleges have higher graduation rates compared to public institutions.
  - **Top10perc** and **Top25perc**: A higher proportion of top-performing students positively impacts graduation rates.
  - **perc.alumni**: Alumni engagement strongly correlates with higher graduation rates.
- **Overfitting**:
  - Ridge and LASSO models showed minimal overfitting, ensuring robustness.

---

## 📜 Full Report

For a detailed analysis, including methodology, regularization plots, and insights, refer to the complete project report:  
[📄 College Graduation Rate Prediction Report](https://github.com/SYEDFAIZAN1987/College-Graduation-Rate-Prediction/blob/main/College%20Graduation%20Rate%20Prediction.pdf)

---

## 📂 Project Structure

```
.
├── Data/
│   ├── College.csv
├── Scripts/
│   ├── College_Graduation_Rate_Prediction.R
├── Reports/
│   ├── College_Graduation_Rate_Prediction.pdf
├── README.md
```
## 🤝 Connect with Me

Feel free to reach out for feedback, questions, or collaboration opportunities:  
**LinkedIn**: [Dr. Syed Faizan](https://www.linkedin.com/in/drsyedfaizanmd/)

---

**Author**: Syed Faizan  
**Master’s Student in Data Analytics and Machine Learning**
