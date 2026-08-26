# 🏦 Loan Approval Prediction using Machine Learning

A machine learning project that predicts whether a customer's loan application is likely to be approved or rejected based on financial and personal information.

## 📌 Project Overview

Loan approval is an important decision-making process in the banking and financial industry. This project uses machine learning to analyze customer information and predict the likelihood of loan approval.

The model uses the following customer details:

* Income
* Loan Amount
* Credit Score
* Age

Based on these features, the system predicts whether the loan will be:

* ✅ Approved
* ❌ Rejected

The project uses **Logistic Regression**, a machine learning algorithm commonly used for binary classification problems.

---

## 🤖 Machine Learning Algorithm

This project uses:

**Logistic Regression**

Logistic Regression is suitable for classification problems where the output belongs to one of two categories.

In this project:

```text
0 = Loan Rejected
1 = Loan Approved
```

The model learns patterns from historical customer data and uses those patterns to predict the loan approval status of new customers.

---

## 📊 Dataset Features

The dataset contains the following columns:

| Feature         | Description                                 |
| --------------- | ------------------------------------------- |
| `income`        | Customer's annual income                    |
| `loan_amount`   | Amount of loan requested                    |
| `credit_score`  | Customer's credit score                     |
| `age`           | Customer's age                              |
| `loan_approved` | Target variable: 0 = Rejected, 1 = Approved |

---

## ⚙️ Project Workflow

```text
Customer Data
      ↓
Data Loading
      ↓
Feature Selection
      ↓
Train/Test Split
      ↓
Logistic Regression Model
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Save Trained Model
      ↓
Predict New Customer Loan Status
```

---

## 📈 Model Evaluation

The model is evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report

These metrics help evaluate how accurately the model predicts loan approval and rejection.

Example output:

```text
Model Accuracy: 1.0

Confusion Matrix:
[[1 0]
 [0 1]]
```

> Note: The dataset used in this project is small and created for learning purposes. A high accuracy score on a very small dataset does not guarantee real-world performance.

---

## 🧠 Model Training

The `main.py` file performs the following tasks:

1. Loads the customer dataset.
2. Separates input features and the target variable.
3. Splits the dataset into training and testing data.
4. Creates a Logistic Regression model.
5. Trains the model using the training data.
6. Predicts loan approval for testing data.
7. Evaluates model performance.
8. Saves the trained model as `loan_model.pkl`.
9. Allows testing with new customer information.

Run the training program:

```bash
py main.py
```

---

## 🔮 Loan Prediction

The `predict.py` file loads the previously trained machine learning model.

It asks the user to enter:

```text
Customer Income
Loan Amount
Credit Score
Customer Age
```

The system then predicts the loan status.

Run:

```bash
py predict.py
```

Example:

```text
Loan Approval Prediction System
-------------------------------

Enter customer income: 60000
Enter loan amount: 200000
Enter credit score: 720
Enter customer age: 40

Loan Status: APPROVED
Approval Probability: 100.00%
```

---

## 📁 Project Structure

```text
loan_prediction/
│
├── .gitignore
├── loan_model.pkl
├── main.py
├── predict.py
├── requirements.txt
└── README.md
```

### File Description

#### `main.py`

This file trains and evaluates the machine learning model.

#### `predict.py`

This file loads the saved model and predicts loan approval for a new customer.

#### `loan_model.pkl`

This is the trained machine learning model saved using `joblib`.

#### `requirements.txt`

This file contains the Python libraries required to run the project.

#### `.gitignore`

This file specifies files and folders that Git should ignore.

---

## 🛠️ Technologies Used

* Python
* Pandas
* Scikit-learn
* Joblib
* Machine Learning
* Logistic Regression

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/misfah2005/loan-prediction.git
```

Navigate to the project folder:

```bash
cd loan-prediction
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train the model

```bash
py main.py
```

### Predict loan approval

```bash
py predict.py
```

Enter the customer's financial information when prompted.

---

## ⚠️ Disclaimer

This project is created for educational and portfolio purposes.

The dataset is small and intended for learning machine learning concepts. This model should not be used for real-world banking or financial decisions without proper validation, larger datasets, fairness testing, security controls, and regulatory review.

---

## 🚀 Future Improvements

Possible future improvements include:

* Using a larger real-world dataset.
* Adding more customer features.
* Data preprocessing and feature scaling.
* Hyperparameter tuning.
* Comparing multiple machine learning algorithms.
* Adding a graphical user interface.
* Creating a web application using Flask or FastAPI.
* Deploying the model online.
* Adding model monitoring and performance tracking.

---

## 👨‍💻 Author

**Mohamed Misfah**

Aspiring AI / Machine Learning Developer

GitHub: https://github.com/misfah2005

---

⭐ If you found this project useful, consider giving the repository a star!
