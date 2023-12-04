from enum import Enum

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from some_code_2 import MyLinearRegression

FILE_NAME = "aparts.csv"


class Cols(Enum):
    PRICE = "price"


class NumCols(Enum):
    SQUARE = "square"
    FLOOR = "floor"
    HOUSE_FLOOR = "house_floor"
    TIME_TO_SUBWAY = "time_to_subway"


class CatCols(Enum):
    ESTATE = "estate"
    DISTRICT = "district"
    REPAIR = "repair"
    HOUSE_TYPE = "house_type"
    SUBWAY = "subway"


def get_cols(class_name):
    return [col.value for col in class_name]


def get_df(file_name: str):
    df = pd.read_table(file_name)
    return df


def clear_data(df: pd.DataFrame):
    df.dropna(subset=["time_to_subway"], inplace=True)
    df["weird"] = df["house_floor"] * df["time_to_subway"]
    num_cols.append("weird")

    lower_quantile, upper_quantile = 0.05, 0.95
    lower_bound = df["price"].quantile(lower_quantile)
    upper_bound = df["price"].quantile(upper_quantile)
    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
    return df


def one_hot_encoding(df: pd.DataFrame, cat_cols: list):
    df = pd.get_dummies(df, columns=cat_cols)

    cat_cols_new = []
    for col_name in cat_cols:
        cat_cols_new.extend(filter(lambda x: x.startswith(col_name), df.columns))
    return df, cat_cols_new


def scale_data(df: pd.DataFrame, num_cols, cat_cols):
    scaler_num = StandardScaler()
    X_num = scaler_num.fit_transform(df[num_cols])
    X_cat = df[cat_cols]
    X = np.concatenate([X_num, X_cat], axis=1)
    return X


def print_metrics(y_preds, y):
    print(f"R^2: {r2_score(y_preds, y)}")
    print(f"MSE: {mean_squared_error(y_preds, y)}")
    print(f"mean_abs_per_error: {mean_absolute_percentage_error(y_preds, y)}")


def print_result(X_test, y_test, models):
    for model in models:
        print(model)
        print_metrics(model.predict(X_test), y_test)
        print()


def fit_models(X_train, y_train):
    mlr = MyLinearRegression()
    mlr.fit(X_train, y_train, epoch=1000, lr=0.9, batch=100)

    sgd = SGDRegressor(learning_rate="constant", tol=0.9, penalty=None)
    sgd.fit(X_train, y_train)

    return mlr, sgd


if __name__ == "__main__":
    num_cols = get_cols(NumCols)
    cat_cols = get_cols(CatCols)
    target_col = Cols.PRICE.value
    cols = num_cols + cat_cols + [target_col]

    data = get_df(FILE_NAME)[cols]
    data = clear_data(data)
    data, cat_cols = one_hot_encoding(data, cat_cols)
    X = scale_data(data, num_cols, cat_cols)
    X_train, X_test, y_train, y_test = train_test_split(
        X, data[target_col], test_size=0.2
    )

    mlr_model, sgd_model = fit_models(X_train, y_train)
    print_result(X_test, y_test, [mlr_model, sgd_model])
