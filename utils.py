import pandas as pd
import datetime as dt

def preprocess_cosumo(data):
    # Add column for total stack bought
    data['STACKS_COMPRATS'] = data['CANTIDADCOMPRA']/data['UNIDADESCONSUMOCONTENIDAS']
    data['STACKS_COMPRATS'] = data['STACKS_COMPRATS'].apply(lambda x: int(x))
    # Separete ORIGIN column with regex
    separated_origin = data['ORIGEN'].str.extract(r'(\d+)\D+(\d+)\D+(\d+)')
    data['REGION'], data['HOSPITAL'], data['DEPARTMENT'] = separated_origin[0], separated_origin[1], separated_origin[2]
    # Drop unnecessary columns
    data.drop(['PRODUCTO', 'NUMERO','ORIGEN','FECHAPEDIDO'], axis = 1, inplace = True)
    # One hot encode the columns
    data_ohe = pd.get_dummies(data, columns=['CODIGO', 'REFERENCIA','TIPOCOMPRA', 'TGL', 'REGION', 'HOSPITAL', 'DEPARTMENT'])
    data_ohe.iloc[:, 6:] = data_ohe.iloc[:, 6:].astype('int')
    print("asjfkjdsfljds")
    return data_ohe

