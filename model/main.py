import os

import numpy as np
import pandas as pd

# import seaborn as sns
# from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression

from statsmodels.tsa.stattools import adfuller

pd.options.mode.chained_assignment = None

ROOT = os.path.dirname(os.path.dirname(__file__))

TARGET_COLUMN = "CLOSE"

DEFAULT_ADF_THRESH = 0.05

MAX_P = 28
MAX_Q = 7

VERBOSE_SAMPLES = 5


def resolve_path(path):
    return ROOT + path


def load_data(data_path):
    data = pd.read_csv(resolve_path(data_path))
    data["DATE"] = pd.to_datetime(data[["YEAR", "MONTH", "DAY"]])
    data.set_index("DATE", inplace=True)
    data = data[[TARGET_COLUMN]]
    return data


def generate_lag_features(data, lag, column=TARGET_COLUMN):
    shifted_columns = pd.DataFrame()
    for i in range(1, lag + 1):
        shifted_columns[f"{column}_SHIFTED_BY_{i}"] = data[column].shift(i)

    return shifted_columns.dropna()


def adf_check(x, thresh=DEFAULT_ADF_THRESH):
    _, p_value, *_ = adfuller(x)
    return p_value <= thresh


def difference(x):
    return x.diff().dropna()


def make_data_stationary(data, column=TARGET_COLUMN, thresh=DEFAULT_ADF_THRESH):
    x = data[column].copy()
    d = 0
    while not adf_check(x):
        x = difference(x)
        d += 1
    return pd.DataFrame(x, columns=[column]), d


def root_mean_squared_error(y, y_hat):
    return np.sqrt(np.mean((y - y_hat) ** 2))


class AR:
    class _Cache:
        def __init__(self):
            self.residuals = None

    def __init__(self, differences):
        self.p = None

        self.coef = None
        self.intercept = None

        self.differences = differences.copy()

        self._cache = AR._Cache()

    def fit(self, verbose=False):
        p = 0
        previous_error = None
        for lag in range(1, MAX_P + 1):
            x = generate_lag_features(self.differences, lag).values
            y = self.differences[lag:].values

            LR = LinearRegression()
            LR.fit(x, y)

            y_hat = LR.predict(x)
            e = root_mean_squared_error(y, y_hat)
            if previous_error is None or e < previous_error:
                p = lag
                previous_error = e

                self.coef = LR.coef_
                self.intercept = LR.intercept_

                self._cache.residuals = y - y_hat

            if verbose:
                print(f"AR:fit() :: p={lag} | error={e}")
                if lag == MAX_P:
                    print(f"At p={p}:")
                    for i in range(VERBOSE_SAMPLES):
                        print("\t", y[i], y_hat[i])
        self.p = p

    def predict(self, x):
        return x * self.coef + self.intercept


class MA:
    def __init__(self, residuals):
        self.q = None

        self.coef = None
        self.intercept = None

        self.residuals = residuals.copy()

    def fit(self, verbose=False):
        q = 0
        previous_error = None
        for lag in range(1, MAX_Q + 1):
            x = generate_lag_features(self.residuals, lag).values
            y = self.residuals[lag:].values

            LR = LinearRegression()
            LR.fit(x, y)

            y_hat = LR.predict(x)
            e = root_mean_squared_error(y, y_hat)
            if previous_error is None or e < previous_error:
                q = lag
                previous_error = e

                self.coef = LR.coef_
                self.intercept = LR.intercept_
            if verbose:
                print(f"MA:fit() :: p={lag} | error={e}")
                if lag == MAX_Q:
                    print(f"At q={q}:")
                    for i in range(VERBOSE_SAMPLES):
                        print("\t", y[i], y_hat[i])
        self.q = q

    def predict(self, x):
        return x * self.coef + self.intercept


class ARIMA:
    class _Cache:
        def __init__(self):
            self.differences = None

    def __init__(self, t_set):
        self.d = None

        self.AR = None
        self.MA = None

        self.t_set = t_set

        self._cache = ARIMA._Cache()

    def fit(self, verbose=False):
        differences, self.d = make_data_stationary(self.t_set)

        self.AR = AR(differences)
        self.AR.fit(verbose)

        residuals = pd.DataFrame(self.AR._cache.residuals, columns=[TARGET_COLUMN])
        self.MA = MA(residuals)
        self.MA.fit(verbose)

        self._cache.differences = differences.values[-self.AR.p - 1 :]

    def forecast(self, steps):
        differences = pd.DataFrame(self._cache.differences, columns=[TARGET_COLUMN])
        x = generate_lag_features(differences, self.AR.p).values

        AR_y_hat = self.AR.predict(x)
        MA_y_hat = self.MA.predict(AR_y_hat)

        print(AR_y_hat)
        print(MA_y_hat)

        pass

    def order(self):
        return self.AR.p, self.d, self.MA.q

    def summary(self):
        print()
        print("=========================")
        print("SUMMARY")
        print("-------------------------")
        print(f"ORDER: {self.order()}")
        print("=========================")
        print()

    def cache(self):
        print()
        print("=========================")
        print(f"CACHE")
        print("-------------------------")
        print(f"DIFFERENCES: {self._cache.differences.shape}")
        print(f"RESIDUALS: {self.AR._cache.residuals.shape}")
        print("=========================")
        print()


if __name__ == "__main__":
    t_set = load_data("/data/PSX/processed/train/data.csv")
    v_set = load_data("/data/PSX/processed/validate/data.csv")

    model = ARIMA(t_set)
    model.fit(verbose=False)

    model.summary()
    model.cache()

    forecasts = model.forecast(steps=31)
