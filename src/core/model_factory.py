from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

from sklearn.linear_model import LinearRegression

from xgboost import XGBRegressor
from catboost import CatBoostRegressor

def build_random_forest(**kwargs):

    defaults = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs)

    return RandomForestRegressor(**defaults)


def build_gradient_boosting(**kwargs):

    defaults = {
        "random_state": 42,
    }

    defaults.update(kwargs)

    return GradientBoostingRegressor(**defaults)


def build_linear_regression():

    return LinearRegression()


def build_extra_trees(**kwargs):

    defaults = {
        "n_estimators": 500,
        "random_state": 42,
        "n_jobs": -1,
    }

    defaults.update(kwargs)

    return ExtraTreesRegressor(**defaults)


def build_xgboost(**kwargs):

    defaults = {

        "n_estimators": 500,

        "learning_rate": 0.05,

        "max_depth": 6,

        "subsample": 0.8,

        "colsample_bytree": 0.8,

        "objective": "reg:squarederror",

        "random_state": 42,

        "n_jobs": -1,
    }

    defaults.update(kwargs)

    return XGBRegressor(**defaults)

def build_catboost(**kwargs):

    defaults = {

        "iterations": 500,

        "learning_rate": 0.05,

        "depth": 6,

        "loss_function": "RMSE",

        "random_seed": 42,

        "verbose": 0,

    }

    defaults.update(kwargs)

    return CatBoostRegressor(**defaults)