from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)

from sklearn.linear_model import LinearRegression


def build_random_forest():

    return RandomForestRegressor(
        n_estimators=1000,
        max_depth=20,
        min_samples_leaf=3,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1,
    )


def build_gradient_boosting():

    return GradientBoostingRegressor(
        random_state=42,
    )


def build_linear_regression():

    return LinearRegression()


def build_extra_trees():

    return ExtraTreesRegressor(
        n_estimators=1000,
        random_state=42,
        n_jobs=-1,
    )