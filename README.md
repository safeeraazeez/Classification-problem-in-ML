# Classification Models Performance Comparison

This project implements and evaluates the performance of seven popular classification algorithms on a dataset. Below is a summary of the models, their accuracies, and recommendations based on the results.

---

## Models and Accuracy

| **Model**               | **Accuracy** |
|--------------------------|--------------|
| Logistic Regression     | 0.982456     |
| Support Vector Machine (SVM) | 0.982456 |
| Random Forest           | 0.964912     |
| Gradient Boosting       | 0.964912     |
| k-Nearest Neighbors (k-NN) | 0.956140 |
| Gaussian Naïve Bayes   | 0.956140     |
| Decision Tree           | 0.921053     |

---

## Key Observations

1. **Best-Performing Models**:
   - **Logistic Regression** and **SVM** achieved the highest accuracy of **98.25%**.
   - These models are particularly suited for the dataset due to its separability and high-dimensional features.

2. **Reliable Performers**:
   - **Random Forest** and **Gradient Boosting** demonstrated robust performance, achieving accuracies of **96.49%**.
   - Ensemble methods like these reduce overfitting and provide consistent results.

3. **Moderate Performers**:
   - **k-NN** and **Gaussian Naïve Bayes** showed slightly lower but respectable accuracies of **95.61%**.

4. **Worst-Performing Model**:
   - **Decision Tree** had the lowest accuracy of **92.11%**, likely due to overfitting or insufficient regularization.

---

## Recommendations

1. **Top Choice**:
   - Use **Logistic Regression** or **SVM** for the best accuracy and performance.

2. **Ensemble Models**:
   - Consider **Random Forest** or **Gradient Boosting** for robust and reliable predictions.

3. **Improvement for Decision Tree**:
   - Apply hyperparameter tuning or regularization to improve Decision Tree performance.

---

## Steps to Reproduce

1. Load the dataset into a Pandas DataFrame.
2. Preprocess the data:
   - Scale the features (StandardScaler is recommended).
   - Remove outliers using the Interquartile Range (IQR) method.
3. Split the data into training and test sets (e.g., 70% training, 30% testing).
4. Train the following models using `scikit-learn`:
   - Logistic Regression
   - Support Vector Machine (SVM)
   - Random Forest
   - Gradient Boosting
   - k-Nearest Neighbors (k-NN)
   - Gaussian Naïve Bayes
   - Decision Tree
5. Evaluate each model’s accuracy on the test set.
6. Create a comparison table and visualize results using Seaborn's barplot.

---

## Visualization

A horizontal bar plot compares the accuracies of all models:
- **X-axis**: Accuracy
- **Y-axis**: Model Names
- Title: "Model Comparison Based on Accuracy"

---

## Tools and Libraries Used

- **Python**: Programming language
- **Pandas**: Data manipulation and analysis
- **scikit-learn**: Machine learning algorithms
- **Seaborn & Matplotlib**: Data visualization

---

## Conclusion

Logistic Regression and SVM are the best choices for this dataset, providing the highest accuracy. Ensemble models (Random Forest and Gradient Boosting) also deliver strong results and are ideal for reducing overfitting. Decision Tree performance can be improved with further tuning.


