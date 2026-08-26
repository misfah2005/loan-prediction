import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = {
    "income": [25000, 35000, 45000, 50000, 60000, 70000, 80000, 90000, 30000, 55000],
    "loan_amount": [100000, 120000, 150000, 180000, 200000, 250000, 300000, 350000, 100000, 220000],
    "credit_score": [580, 620, 680, 700, 720, 750, 780, 800, 600, 710],
    "age": [25, 30, 35, 40, 45, 50, 55, 60, 28, 42],
    "loan_approved": [0, 0, 1, 1, 1, 1, 1, 1, 0, 1]
}

df = pd.DataFrame(data)

print(df)

X = df[["income", "loan_amount", "credit_score", "age"]]

y = df["loan_approved"]

print("\nFeatures (X):")
print(X)

print("\nTarget (y):")
print(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("\nModel trained successfully!")

joblib.dump(model, "loan_model.pkl")

print("Model saved successfully!")

predictions = model.predict(X_test)

print("\nPredicted loan approval:")
print(predictions)

print("\nActual loan approval:")
print(y_test.values)

accuracy = accuracy_score(y_test, predictions)

print("\nModel Accuracy:")
print(accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

print("\nTraining data:")
print(X_train)

print("\nTesting data:")
print(X_test)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))

print("\nLoan Approval Prediction System")
print("-------------------------------")

income = float(input("Enter customer income: "))
loan_amount = float(input("Enter loan amount: "))
credit_score = float(input("Enter credit score: "))
age = float(input("Enter customer age: "))

new_customer = pd.DataFrame(
    [[income, loan_amount, credit_score, age]],
    columns=["income", "loan_amount", "credit_score", "age"]
)

prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

approval_probability = probability[0][1] * 100
rejection_probability = probability[0][0] * 100

if prediction[0] == 1:
    print("\nLoan Status: APPROVED")
    print(f"Approval Probability: {approval_probability:.2f}%")
else:
    print("\nLoan Status: REJECTED")
    print(f"Rejection Probability: {rejection_probability:.2f}%")