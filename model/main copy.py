import os

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.stattools import adfuller

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

pd.options.mode.chained_assignment = None

ROOT = os.path.dirname(os.path.dirname(__file__))

TARGET_COLUMN = "CLOSE"
ADF_THRESH = 0.05


def resolve_path(path):
    return ROOT + path


def load_data(data_path):
    data = pd.read_csv(resolve_path(data_path))
    data["DATE"] = pd.to_datetime(data[["YEAR", "MONTH", "DAY"]])
    data.set_index("DATE", inplace=True)
    data = data[[TARGET_COLUMN]]
    return data


def apply_lag_shift(data, column, lag):
    shifted_columns = pd.DataFrame()
    for i in range(1, lag + 1):
        shifted_columns[f"{column}_SHIFTED_BY_{i}"] = data[column].shift(i)
    return shifted_columns.dropna()


def adf_check(X, thresh=ADF_THRESH):
    _, p_value, *_ = adfuller(X)
    return p_value <= thresh


def difference(X):
    return np.log(X).diff().dropna()


def make_data_stationary(data, column, thresh=ADF_THRESH):
    X = data[column].copy()
    while True:
        _, p_value, *_ = adfuller(X)
        if p_value >= thresh:
            X = np.log(X).diff().dropna()
        else:
            break
    return X


class AR:
    def __init__(self, s_data_with_lag):
        self.s_data = s_data_with_lag.copy()

    def fit(self):
        pass

    def predict(self):
        pass


class MA:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class ARIMA:
    def __init__(self, data, split):
        self.p = None
        self.d = None
        self.q = None

        self.df = data
        self.theta_AR = None
        self.intercept_AR = None
        self.theta_MA = None
        self.intercept_MA = None
        self.df_c = None
        self.df_train = None
        self.split = split
        self.df_diff = make_data_stationary(data, TARGET_COLUMN)

    def AR(self, p, df):
        df_temp = df.copy()

        # Generating the lagged p terms
        for i in range(1, p + 1):
            df_temp["Shifted_values_%d" % i] = df_temp["Value"].shift(i)

        train_size = (int)(self.split * df_temp.shape[0])

        df_train = df_temp.iloc[:train_size]
        df_test = df_temp.iloc[train_size:]

        df_train_2 = df_train.dropna()
        # X contains the lagged values ,hence we skip the first column
        X_train = df_train_2.iloc[:, 1:].values.reshape(-1, p)
        # Y contains the value,it is the first column
        y_train = df_train_2.iloc[:, 0].values.reshape(-1, 1)

        # Running linear regression to generate the coefficents of lagged terms
        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        df_train_2["Predicted_Values"] = X_train.dot(lr.coef_.T) + lr.intercept_
        # df_train_2[['Value','Predicted_Values']].plot()

        X_test = df_test.iloc[:, 1:].values.reshape(-1, p)
        df_test["Predicted_Values"] = X_test.dot(lr.coef_.T) + lr.intercept_
        # df_test[['Value','Predicted_Values']].plot()

        RMSE = np.sqrt(
            mean_squared_error(df_test["Value"], df_test["Predicted_Values"])
        )

        print("The RMSE is :", RMSE, ", Value of p : ", p)
        return [df_train_2, df_test, theta, intercept, RMSE]

    def MA(self, q, res):
        for i in range(1, q + 1):
            res["Shifted_values_%d" % i] = res["Residuals"].shift(i)

        train_size = (int)(self.split * res.shape[0])

        res_train = res.iloc[:train_size]
        res_test = res.iloc[train_size:]

        res_train_2 = res_train.dropna()
        X_train = res_train_2.iloc[:, 1:].values.reshape(-1, q)
        y_train = res_train_2.iloc[:, 0].values.reshape(-1, 1)

        lr = LinearRegression()
        lr.fit(X_train, y_train)

        theta = lr.coef_.T
        intercept = lr.intercept_
        res_train_2["Predicted_Values"] = X_train.dot(lr.coef_.T) + lr.intercept_
        # res_train_2[['Residuals','Predicted_Values']].plot()

        X_test = res_test.iloc[:, 1:].values.reshape(-1, q)
        res_test["Predicted_Values"] = X_test.dot(lr.coef_.T) + lr.intercept_
        # res_test[['Residuals','Predicted_Values']].plot()

        RMSE = np.sqrt(
            mean_squared_error(res_test["Residuals"], res_test["Predicted_Values"])
        )

        print("The RMSE is :", RMSE, ", Value of q : ", q)
        return [res_train_2, res_test, theta, intercept, RMSE]

    def fit(self):
        # Ensure df_testing is a DataFrame with a column named 'Value'
        self.df_diff = pd.DataFrame(self.df_diff, columns=["Value"])

        best_RMSE = 100000000000
        best_p = -1

        for i in range(1, 21):
            [df_train, df_test, theta, intercept, RMSE] = self.AR(
                i, pd.DataFrame(self.df_diff.Value)
            )
            if RMSE < best_RMSE:
                best_RMSE = RMSE
                best_p = i
                self.p = best_p
                self.theta_AR = theta  # Update theta_AR
                self.intercept_AR = intercept  # Update intercept_AR

        print("Best value for p:", best_p)

        [self.df_train, df_test, theta, intercept, RMSE] = self.AR(
            best_p, pd.DataFrame(self.df_diff.Value)
        )
        self.df_c = pd.concat([self.df_train, df_test])

        res = pd.DataFrame()
        res["Residuals"] = self.df_c.Value - self.df_c.Predicted_Values

        best_RMSE = 100000000000
        best_q = -1

        for i in range(1, 13):
            [res_train, res_test, theta, intercept, RMSE] = self.MA(
                i, pd.DataFrame(res.Residuals)
            )
            if RMSE < best_RMSE:
                best_RMSE = RMSE
                best_q = i
                self.q = best_q
                self.theta_MA = theta  # Update theta_MA
                self.intercept_MA = intercept  # Update intercept_MA

        print("Best value for q:", best_q)

        [res_train, res_test, theta, intercept, RMSE] = self.MA(
            best_q, pd.DataFrame(res.Residuals)
        )
        # print(theta)
        # print(intercept)

        res_c = pd.concat([res_train, res_test])
        self.df_c.Predicted_Values += res_c.Predicted_Values

        # Revert transformations to obtain the actual 'CLOSE' values
        # Convert the 'Value' column in 'df' to numeric type if it contains dates
        self.df["Value"] = pd.to_numeric(df["Value"], errors="coerce")

        # Extract numeric values from 'Value' column for log transformation
        numeric_values = df["Value"][
            pd.to_numeric(df["Value"], errors="coerce").notna()
        ]

        # Use index-based access instead of '.Value'
        self.df_c["Value"] += np.log(numeric_values).shift(1)
        self.df_c["Value"] += np.log(numeric_values).diff().shift(12)
        self.df_c["Predicted_Values"] += np.log(numeric_values).shift(1)
        self.df_c["Predicted_Values"] += np.log(numeric_values).diff().shift(12)

        self.df_c["Value"] = np.exp(self.df_c["Value"])
        self.df_c["Predicted_Values"] = np.exp(self.df_c["Predicted_Values"])


if __name__ == "__main__":
    training_set = load_data("/data/PSX/processed/train/data.csv")
    validation_set = load_data("/data/PSX/processed/validate/data.csv")

    print(adf_check(validation_set[TARGET_COLUMN].values))
    X = make_data_stationary(validation_set, TARGET_COLUMN)
    print(adf_check(X))

    shifted_columns = apply_lag_shift(validation_set, TARGET_COLUMN, 19)
    shifted_columns.info()

    # model = ARIMA(training_set)
    # model.fit()

    # model.summary()

    # model.validate(validation_set)
