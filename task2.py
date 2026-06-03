# Task 2: Credit Risk Prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    print("--- Task 2: Credit Risk Prediction ---")
    
    # 1. Create Synthetic Data (Mimicking a Loan Dataset)
    # In a real scenario, you would load: df = pd.read_csv('loan_data.csv')
    data = {
        'ApplicantIncome': np.random.randint(2000, 10000, 500),
        'LoanAmount': (np.random.rand(500) * 200 + 50).round(2),
        'Education': np.random.choice(['Graduate', 'Not Graduate'], 500),
        'Credit_History': np.random.choice([0.0, 1.0], 500, p=[0.15, 0.85]),
        # Target variable (1 = Default/No, 0 = Approved/Yes - Logic varies, here 0=Good, 1=Bad)
        'Loan_Status': np.random.choice([0, 1], 500) 
    }
    df = pd.DataFrame(data)

    # Introduce some missing values to demonstrate handling skills
    df.loc[10:15, 'LoanAmount'] = np.nan
    df.loc[20:22, 'Credit_History'] = np.nan

    print("\nOriginal Data Head:")
    print(df.head())

    # 2. Handle Missing Data Appropriately
    # Fill numerical missing values with the mean
    df['LoanAmount'].fillna(df['LoanAmount'].mean(), inplace=True)
    # Fill categorical missing values with the mode
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    print("\nMissing Values after handling:")
    print(df.isnull().sum())

    # 3. Encode Categorical Features
    # Convert 'Education' to binary (0 and 1)
    df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})

    # 4. Visualize Key Features
    plt.figure(figsize=(10, 5))
    sns.countplot(x='Education', hue='Loan_Status', data=df)
    plt.title('Loan Status by Education')
    plt.savefig('task2_education_viz.png')
    plt.show()

    # 5. Train Classification Model
    # Define features (X) and target (y)
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']

    # Split into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train Logistic Regression
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    predictions = model.predict(X_test)
    
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    main()