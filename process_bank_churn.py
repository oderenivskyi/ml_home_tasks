import pandas as pd
from pandas import DataFrame, Series
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from typing import Tuple, Dict, Union, Optional


def split_features_targets(raw_df: DataFrame, target_col: str, drop_cols: list) -> Tuple[DataFrame, Series, DataFrame, Series, list]:
    """
    Розділяє датасет на навчальні та валідаційні ознаки і цільову змінну.
    """
    input_cols = raw_df.drop(columns=drop_cols).columns.to_list()
    train_X, val_X, train_y, val_y = train_test_split(
        raw_df[input_cols], raw_df[target_col], test_size=0.2, random_state=17, stratify = raw_df[target_col]
    )
    return train_X, val_X, train_y, val_y, input_cols


def impute_data(
    train_df: DataFrame, val_df: DataFrame
) -> Tuple[DataFrame, DataFrame, list, list, SimpleImputer, SimpleImputer]:
    """
    Застосовує імпутацію до числових і категоріальних ознак.
    """
    numeric_cols = train_df.select_dtypes(include='number').columns.tolist()
    categorical_cols = train_df.select_dtypes(include='object').columns.tolist()

    num_imputer = SimpleImputer(strategy='mean')
    cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

    train_df[numeric_cols] = num_imputer.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = num_imputer.transform(val_df[numeric_cols])

    train_df[categorical_cols] = cat_imputer.fit_transform(train_df[categorical_cols])
    val_df[categorical_cols] = cat_imputer.transform(val_df[categorical_cols])

    return train_df, val_df, numeric_cols, categorical_cols, num_imputer, cat_imputer


def scale_numeric_data(
    train_df: DataFrame, val_df: DataFrame, numeric_cols: list
) -> Tuple[DataFrame, DataFrame, MinMaxScaler]:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler.
    """
    scaler = MinMaxScaler()
    train_df[numeric_cols] = scaler.fit_transform(train_df[numeric_cols])
    val_df[numeric_cols] = scaler.transform(val_df[numeric_cols])
    return train_df, val_df, scaler


def encode_categorical_data(
    train_df: DataFrame, val_df: DataFrame, categorical_cols: list
) -> Tuple[DataFrame, DataFrame, list, OneHotEncoder]:
    """
    Кодує категоріальні ознаки за допомогою OneHotEncoder.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train_df[categorical_cols])
    encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))

    train_encoded = encoder.transform(train_df[categorical_cols])
    val_encoded = encoder.transform(val_df[categorical_cols])

    train_df[encoded_col_names] = train_encoded
    val_df[encoded_col_names] = val_encoded

    return train_df, val_df, encoded_col_names, encoder


def preprocess_data(
    raw_df: DataFrame, scale_numeric: bool = True
) -> Dict[str, Union[DataFrame, Series, list, MinMaxScaler, OneHotEncoder]]:
    """
    Повна обробка навчальних і валідаційних даних: імпутація, масштабування, кодування.
    """
    train_X, val_X, train_y, val_y, input_cols = split_features_targets(
        raw_df, target_col='Exited', drop_cols=['RowNumber', 'CustomerId', 'Surname', 'Exited']
    )

    train_X, val_X, numeric_cols, categorical_cols, num_imputer, cat_imputer = impute_data(train_X, val_X)

    scaler = None
    if scale_numeric:
        train_X, val_X, scaler = scale_numeric_data(train_X, val_X, numeric_cols)

    train_X, val_X, encoded_cols, encoder = encode_categorical_data(train_X, val_X, categorical_cols)

    X_train = train_X[numeric_cols + encoded_cols]
    X_val = val_X[numeric_cols + encoded_cols]

    return {
        'X_train': X_train,
        'train_targets': train_y,
        'X_val': X_val,
        'val_targets': val_y,
        'input_cols': input_cols,
        'numeric_cols': numeric_cols,
        'categorical_cols': categorical_cols,
        'scaler': scaler,
        'encoder': encoder,
        'num_imputer': num_imputer,
        'cat_imputer': cat_imputer
    }


def preprocess_new_data(
    new_data: DataFrame,
    numeric_cols: list,
    categorical_cols: list,
    num_imputer: SimpleImputer,
    cat_imputer: SimpleImputer,
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> DataFrame:
    """
    Обробка нових даних на основі вже навчених імпутера, скейлера та енкодера.
    """
    new_data = new_data.copy()

    new_data[numeric_cols] = num_imputer.transform(new_data[numeric_cols])
    new_data[categorical_cols] = cat_imputer.transform(new_data[categorical_cols])

    if scaler:
        new_data[numeric_cols] = scaler.transform(new_data[numeric_cols])

    encoded_col_names = list(encoder.get_feature_names_out(categorical_cols))
    encoded_data = encoder.transform(new_data[categorical_cols])
    new_data[encoded_col_names] = encoded_data

    processed_data = new_data[numeric_cols + encoded_col_names]

    return processed_data
