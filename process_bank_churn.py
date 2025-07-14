from typing import Dict, List, Union, Optional
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder


def split_features_targets(df: pd.DataFrame) -> tuple:
    """
    Видаляє службові колонки, ділить датафрейм на ознаки та цільову змінну.
    """
    input_cols = df.drop(['RowNumber', 'CustomerId', 'Surname', 'Exited'], axis=1).columns.to_list()
    X = df[input_cols]
    y = df['Exited']
    return X, y, input_cols


def split_train_val(X: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Ділить ознаки та ціль на тренувальні та валідаційні набори.
    """
    return train_test_split(X, y, test_size=0.2, random_state=17)


def impute_numeric_data(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple:
    """
    Імпутує числові стовпці середнім значенням.

    Returns:
        Оброблені дані, список числових колонок і numeric_imputer.
    """
    numeric_cols = X_train.select_dtypes(include='number').columns.tolist()
    numeric_imputer = SimpleImputer(strategy='mean')
    X_train[numeric_cols] = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = numeric_imputer.transform(X_val[numeric_cols])
    return X_train, X_val, numeric_cols, numeric_imputer


def impute_categorical_data(X_train: pd.DataFrame, X_val: pd.DataFrame) -> tuple:
    """
    Імпутує категоріальні стовпці значенням 'Unknown'.

    Returns:
        Оброблені дані, список категоріальних колонок і categorical_imputer.
    """
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()
    categorical_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')
    X_train[categorical_cols] = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = categorical_imputer.transform(X_val[categorical_cols])
    return X_train, X_val, categorical_cols, categorical_imputer


def scale_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    numeric_cols: List[str],
    do_scale: bool
) -> tuple:
    """
    Масштабує числові ознаки за допомогою MinMaxScaler, якщо увімкнено.
    """
    scaler = None
    if do_scale:
        scaler = MinMaxScaler()
        X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
        X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    return X_train, X_val, scaler


def encode_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    categorical_cols: List[str]
) -> tuple:
    """
    Виконує one-hot енкодинг для категоріальних ознак.
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(X_train[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols).tolist()
    X_train[encoded_cols] = encoder.transform(X_train[categorical_cols])
    X_val[encoded_cols] = encoder.transform(X_val[categorical_cols])
    return X_train, X_val, encoder, encoded_cols


def preprocess_data(raw_df: pd.DataFrame, scaler_numeric: bool) -> Dict[str, Union[pd.DataFrame, pd.Series, List[str], MinMaxScaler, OneHotEncoder]]:
    """
    Основна функція обробки даних: очищення, імпутація, масштабування, кодування.
    """
    X, y, input_cols = split_features_targets(raw_df)
    X_train, X_val, y_train, y_val = split_train_val(X, y)

    X_train, X_val, numeric_cols, numeric_imputer = impute_numeric_data(X_train, X_val)
    X_train, X_val, categorical_cols, categorical_imputer = impute_categorical_data(X_train, X_val)

    X_train, X_val, scaler = scale_data(X_train, X_val, numeric_cols, scaler_numeric)
    X_train, X_val, encoder, encoded_cols = encode_data(X_train, X_val, categorical_cols)

    return {
        'X_train': X_train,
        'train_targets': y_train,
        'X_val': X_val,
        'val_targets': y_val,
        'input_cols': input_cols,
        'scaler': scaler,
        'encoder': encoder,
    }


def preprocess_new_data(
    new_df: pd.DataFrame,
    input_cols: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    numeric_imputer: SimpleImputer,
    categorical_imputer: SimpleImputer,
    scaler: Optional[MinMaxScaler],
    encoder: OneHotEncoder
) -> pd.DataFrame:
    """
    Обробляє нові дані з використанням навчених імпутерів, скейлера і енкодера.
    """
    data = new_df[input_cols].copy()

    data[numeric_cols] = numeric_imputer.transform(data[numeric_cols])
    data[categorical_cols] = categorical_imputer.transform(data[categorical_cols])

    if scaler is not None:
        data[numeric_cols] = scaler.transform(data[numeric_cols])

    encoded = encoder.transform(data[categorical_cols])
    encoded_cols = encoder.get_feature_names_out(categorical_cols)
    data[encoded_cols] = encoded

    return data.drop(columns=categorical_cols)
