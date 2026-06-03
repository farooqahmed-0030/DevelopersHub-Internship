# Task 1: Exploring and Visualizing a Simple Dataset
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def main():
    # 1. Load the dataset using seaborn
    # Seaborn has built-in datasets, so we don't need an external CSV file for this task.
    iris = sns.load_dataset('iris')
    
    print("--- Task 1: Iris Dataset Analysis ---")

    # 2. Display dataset structure
    print("\nDataset Shape:")
    print(iris.shape)
    
    print("\nDataset Columns:")
    print(iris.columns)
    
    print("\nFirst 5 Rows (Head):")
    print(iris.head())

    # 3. Create Basic Visualizations
    # Set a generic style for the plots
    sns.set_theme(style="whitegrid")

    # A. Scatterplot to analyze relationships between variables
    # Analyzing relationship between Sepal Length and Sepal Width, colored by species
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sepal_length', y='sepal_width', hue='species', data=iris, palette='deep')
    plt.title('Scatterplot: Sepal Length vs Sepal Width')
    plt.savefig('task1_scatterplot.png') # Saving the figure
    plt.show()

    # B. Histogram to examine data distribution
    # Analyzing the distribution of Petal Length
    plt.figure(figsize=(10, 6))
    sns.histplot(iris['petal_length'], kde=True, bins=20, color='purple')
    plt.title('Histogram: Distribution of Petal Length')
    plt.savefig('task1_histogram.png')
    plt.show()

    # C. Box plot to detect outliers and spread of values
    # Analyzing spread of all numerical features
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=iris)
    plt.title('Boxplot: Spread of Values (Detection of Outliers)')
    plt.savefig('task1_boxplot.png')
    plt.show()

    print("Visualizations saved successfully.")

if __name__ == "__main__":
    main()