# Import libraries
import pandas as pd
import pickle
import sys
from numpy import savetxt
sys.path.insert(1, '../utils')
from preprocessing import preprocess_cosumo


# Read dataset
df = pd.read_csv('data/consumo_material_clean_with_category.csv', parse_dates=['FECHAPEDIDO'])

# Load models
xgb_transito_models = [pickle.load(open(f'trained_models/xgb_transito_{i}Q.pkl', 'rb')) for i in range(1, 5)]
xgb_specified_ids_models = [pickle.load(open(f'trained_models/xgb_specified_ids_{i}Q.pkl', 'rb')) for i in range(1, 5)]
xgb_other_ids_lt_10_models = [pickle.load(open(f'trained_models/xgb_other_ids_lt_10_{i}Q.pkl', 'rb')) for i in range(1, 5)]
xgb_other_ids_eq_10_models = [pickle.load(open(f'trained_models/xgb_other_ids_eq_10_{i}Q.pkl', 'rb')) for i in range(1, 5)]
xgb_other_ids_gt_10_models = [pickle.load(open(f'trained_models/xgb_other_ids_gt_10_{i}Q.pkl', 'rb')) for i in range(1, 5)]

# Make data separations using rules found by previous data analysis
df_transito = df[df['TGL'] == 'TRANSITO']
df_alma = df[df['TGL'] == 'ALMACENABLE']

specified_ids = ['E65159', 'E64751', 'E64764']
df_specified_ids = df_alma[df_alma['CODIGO'].isin(specified_ids)]
df_other_ids = df_alma[~df_alma['CODIGO'].isin(specified_ids)]
df_other_ids_gt_10 = df_other_ids[df_other_ids['UNIDADESCONSUMOCONTENIDAS'] > 10]
df_other_ids_eq_10 = df_other_ids[df_other_ids['UNIDADESCONSUMOCONTENIDAS'] == 10]
df_other_ids_lt_10 = df_other_ids[df_other_ids['UNIDADESCONSUMOCONTENIDAS'] < 10]

# Initialitze list to store the results and apply all the models
predictions_2023F = []
gt_2023F = []

# Function to apply each of the trained models to each of the data division
def apply_xgboost_models(df_test, xgb_model):
    predictions_2023 = []
    gt_2023 = []
    df_test.drop(['trimester'], axis = 1, inplace = True)
    df_test.drop(columns=['FECHAPEDIDO'], inplace=True)
    X_test = df_test.drop(['STACKS_COMPRATS'], axis = 1)
    y_test = df_test['STACKS_COMPRATS'].reset_index(drop=True)
    y_predicted_test = xgb_model.predict(X_test)
    predictions_2023.append(y_predicted_test)
    gt_2023.append(y_test)
    return gt_2023, predictions_2023

# Iterate through the different dataset, group them by trimestre (quarter-year) + preprocess it + do the prediciton using the previous function
for df_name, df in [('df_transito', df_transito), ('df_specified_ids', df_specified_ids), ('df_other_ids_gt_10', df_other_ids_gt_10), ('df_other_ids_eq_10', df_other_ids_eq_10), ('df_other_ids_lt_10', df_other_ids_lt_10)]:
    df = preprocess_cosumo(df)
    df = df[df['FECHAPEDIDO'].dt.year == 2023]
    df['trimester'] = df['FECHAPEDIDO'].dt.quarter
    grouped_df = df.groupby(['trimester', df['FECHAPEDIDO'].dt.year])
    trimester_dfs = [grouped_df.get_group((trimester, year)).reset_index(drop=True) for (trimester, year), _ in grouped_df]
    result_dfs = [trimester_dfs[i] for i in range(0, len(trimester_dfs))]
    
    # Apply the trained models
    for i, result_df in enumerate(result_dfs):
        if df_name == 'df_transito':
            print(df_name)
            a, b = apply_xgboost_models(result_df, xgb_transito_models[i])
            gt_2023F += a
            predictions_2023F += b
        elif df_name == 'df_specified_ids':
            print(df_name)
            a, b = apply_xgboost_models(result_df, xgb_specified_ids_models[i])
            gt_2023F += a
            predictions_2023F += b
        elif df_name == 'df_other_ids_lt_10':
            print(df_name)
            a, b = apply_xgboost_models(result_df, xgb_other_ids_lt_10_models[i])
            gt_2023F += a
            predictions_2023F += b
        elif df_name == 'df_other_ids_eq_10':
            print(df_name)
            a, b = apply_xgboost_models(result_df, xgb_other_ids_eq_10_models[i])
            gt_2023F += a
            predictions_2023F += b
        elif df_name == 'df_other_ids_gt_10':
            print(df_name)
            a, b = apply_xgboost_models(result_df, xgb_other_ids_gt_10_models[i])
            gt_2023F += a
            predictions_2023F += b
    

# Concatenate the prediction of the whole dataset
pred = np.concatenate(predictions_2023F).flatten()
gt = np.concatenate(gt_2023F).flatten()

# Save the data
savetxt('predictions.csv', pred, delimiter=',')
