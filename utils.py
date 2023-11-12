import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def preprocess_cosumo(data):
    # Add column for total stack bought
    data['STACKS_COMPRATS'] = data['CANTIDADCOMPRA']/data['UNIDADESCONSUMOCONTENIDAS']
    data['STACKS_COMPRATS'] = data['STACKS_COMPRATS'].apply(lambda x: int(x))
    # Separete ORIGIN column with regex
    separated_origin = data['ORIGEN'].str.extract(r'(\d+)\D+(\d+)\D+(\d+)')
    data['REGION'], data['HOSPITAL'], data['DEPARTMENT'] = separated_origin[0], separated_origin[1], separated_origin[2]
    # Drop unnecessary columns
    data.drop(['PRODUCTO', 'NUMERO','ORIGEN','CANTIDADCOMPRA','IMPORTELINEA', 'REFERENCIA', 'REGION', 'HOSPITAL','DEPARTMENT', 'CATEGORY'], axis = 1, inplace = True)
    # One hot encode the columns
    data_ohe = pd.get_dummies(data, columns=['CODIGO','TIPOCOMPRA'])
    data_ohe.iloc[:, 5:] = data_ohe.iloc[:, 5:].astype('int')
    #weight_mapping = {'ALMACENABLE': 1, 'TRANSITO': 0} 
    #data_ohe['TGL_weighted'] = data_ohe['TGL'].map(weight_mapping)
    data_ohe = data_ohe.drop(columns=['TGL'])
    print("asjfkjdsfljds")
    return data_ohe

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
    plt.axhline(y=0, color='red', linestyle='--')  # Add a horizontal line at y=0 for reference
    plt.title('Residuals Scatter Plot')
    plt.xlabel('True Values')
    plt.ylabel('Residuals')
    plt.show()

