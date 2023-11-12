# Import libraries
import matplotlib.pyplot as plt
import numpy as np

def residuals_hist(y_true, y_pred):
    residuals = np.abs(y_true - y_pred)
    # Create a histogram of residuals
    plt.hist(residuals, bins=50, edgecolor='black')
    plt.title('Distribution of Residuals')
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.show()

def residuals_scatter(y_true, y_pred):
    residuals = np.abs(y_true - y_pred)
    # Alternatively, create a scatter plot
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title('Residuals Scatter Plot')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()