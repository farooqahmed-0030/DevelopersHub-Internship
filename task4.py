# Task 4: Predicting Insurance Claim Amounts
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def main():
    print("--- Task 4: Predicting Insurance Claim Amounts ---")

    # 1. Create Synthetic Data (Mimicking Medical Cost Personal Dataset)
    np.random.seed(42)
    n_samples = 400
    
    data = {
        'age': np.random.randint(18, 70, n_samples),
        'bmi': np.random.uniform(18, 40, n_samples),
        'smoker': np.random.choice(['yes', 'no'], n_samples),
        # Charges are heavily influenced by Age, BMI, and Smoking status
        'charges': np.random.uniform(2000, 50000, n_samples) 
    }
    df = pd.DataFrame(data)

    # Add logic to make 'charges' dependent on features so the model actually learns something
    # If smoker, charges are higher
    smoker_penalty = df['smoker'].apply(lambda x: 20000 if x == 'yes' else 0)
    df['charges'] = (df['age'] * 300) + (df['bmi'] * 400) + smoker_penalty + np.random.normal(0, 2000, n_samples)

    print("\nData Sample:")
    print(df.head())

    # 2. Encode Categorical Features (Smoker status)
    # Convert 'smoker' yes/no to 1/0
    df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})

    # 3. Train Linear Regression Model
    X = df[['age', 'bmi', 'smoker']]
    y = df['charges']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # 4. Evaluate Model Performance (MAE and RMSE)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # 5. Visualize Impact of Features on Charges
    
    # Visualization 1: Impact of BMI and Charges colored by Smoker
    plt.figure(figsize=(10, 6))
    # We reconstruct a temporary dataframe for plotting purposes using original boolean mapping
    plot_df = X_test.copy()
    plot_df['charges'] = y_test
    plot_df['Smoker_Status'] = plot_df['smoker'].map({1: 'Smoker', 0: 'Non-Smoker'})
    
    sns.scatterplot(x='bmi', y='charges', hue='Smoker_Status', data=plot_df, alpha=0.7)
    plt.title('Impact of BMI on Insurance Charges (by Smoker Status)')
    plt.savefig('task4_bmi_impact.png')
    plt.show()

    # Visualization 2: Actual vs Predicted Charges
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, color='blue')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2) # Diagonal line
    plt.xlabel('Actual Charges')
    plt.ylabel('Predicted Charges')
    plt.title('Actual vs Predicted Insurance Charges')
    plt.savefig('task4_prediction_accuracy.png')
    plt.show()

if __name__ == "__main__":
    main()