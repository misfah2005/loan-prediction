import pandas as pd
import joblib


# Load trained model
model = joblib.load("loan_model.pkl")


print("Loan Approval Prediction System")
print("-------------------------------")


# Get customer details
income = float(input("Enter customer income: "))
loan_amount = float(input("Enter loan amount: "))
credit_score = float(input("Enter credit score: "))
age = float(input("Enter customer age: "))


# Create input DataFrame
new_customer = pd.DataFrame(
    [[income, loan_amount, credit_score, age]],
    columns=["income", "loan_amount", "credit_score", "age"]
)


# Make prediction
prediction = model.predict(new_customer)

# Get probabilities
probability = model.predict_proba(new_customer)

approval_probability = probability[0][1] * 100
rejection_probability = probability[0][0] * 100


# Display result
if prediction[0] == 1:
    print("\nLoan Status: APPROVED")
    print(f"Approval Probability: {approval_probability:.2f}%")
else:
    print("\nLoan Status: REJECTED")
    print(f"Rejection Probability: {rejection_probability:.2f}%")