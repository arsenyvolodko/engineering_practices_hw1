--- some_code_1.py	2023-12-04 11:34:45.300341+00:00
+++ some_code_1.py	2023-12-04 11:34:54.369011+00:00
@@ -7,30 +7,30 @@
 from sklearn.metrics import mean_squared_error, r2_score
 from sklearn.metrics import mean_absolute_percentage_error
 
 from some_code_2 import MyLinearRegression
 
-FILE_NAME = 'aparts.csv'
+FILE_NAME = "aparts.csv"
 
 
 class Cols(Enum):
-    PRICE = 'price'
+    PRICE = "price"
 
 
 class NumCols(Enum):
-    SQUARE = 'square'
-    FLOOR = 'floor'
-    HOUSE_FLOOR = 'house_floor'
-    TIME_TO_SUBWAY = 'time_to_subway'
+    SQUARE = "square"
+    FLOOR = "floor"
+    HOUSE_FLOOR = "house_floor"
+    TIME_TO_SUBWAY = "time_to_subway"
 
 
 class CatCols(Enum):
-    ESTATE = 'estate'
-    DISTRICT = 'district'
-    REPAIR = 'repair'
-    HOUSE_TYPE = 'house_type'
-    SUBWAY = 'subway'
+    ESTATE = "estate"
+    DISTRICT = "district"
+    REPAIR = "repair"
+    HOUSE_TYPE = "house_type"
+    SUBWAY = "subway"
 
 
 def get_cols(class_name):
     return [col.value for col in class_name]
 
@@ -39,18 +39,18 @@
     df = pd.read_table(file_name)
     return df
 
 
 def clear_data(df: pd.DataFrame):
-    df.dropna(subset=['time_to_subway'], inplace=True)
-    df['weird'] = df['house_floor'] * df['time_to_subway']
-    num_cols.append('weird')
+    df.dropna(subset=["time_to_subway"], inplace=True)
+    df["weird"] = df["house_floor"] * df["time_to_subway"]
+    num_cols.append("weird")
 
     lower_quantile, upper_quantile = 0.05, 0.95
-    lower_bound = df['price'].quantile(lower_quantile)
-    upper_bound = df['price'].quantile(upper_quantile)
-    df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
+    lower_bound = df["price"].quantile(lower_quantile)
+    upper_bound = df["price"].quantile(upper_quantile)
+    df = df[(df["price"] >= lower_bound) & (df["price"] <= upper_bound)]
     return df
 
 
 def one_hot_encoding(df: pd.DataFrame, cat_cols: list):
     df = pd.get_dummies(df, columns=cat_cols)
@@ -68,13 +68,13 @@
     X = np.concatenate([X_num, X_cat], axis=1)
     return X
 
 
 def print_metrics(y_preds, y):
-    print(f'R^2: {r2_score(y_preds, y)}')
-    print(f'MSE: {mean_squared_error(y_preds, y)}')
-    print(f'mean_abs_per_error: {mean_absolute_percentage_error(y_preds, y)}')
+    print(f"R^2: {r2_score(y_preds, y)}")
+    print(f"MSE: {mean_squared_error(y_preds, y)}")
+    print(f"mean_abs_per_error: {mean_absolute_percentage_error(y_preds, y)}")
 
 
 def print_result(X_test, y_test, models):
     for model in models:
         print(model)
@@ -84,25 +84,27 @@
 
 def fit_models(X_train, y_train):
     mlr = MyLinearRegression()
     mlr.fit(X_train, y_train, epoch=1000, lr=0.9, batch=100)
 
-    sgd = SGDRegressor(learning_rate='constant', tol=0.9, penalty=None)
+    sgd = SGDRegressor(learning_rate="constant", tol=0.9, penalty=None)
     sgd.fit(X_train, y_train)
 
     return mlr, sgd
 
 
-if __name__ == '__main__':
+if __name__ == "__main__":
     num_cols = get_cols(NumCols)
     cat_cols = get_cols(CatCols)
     target_col = Cols.PRICE.value
     cols = num_cols + cat_cols + [target_col]
 
     data = get_df(FILE_NAME)[cols]
     data = clear_data(data)
     data, cat_cols = one_hot_encoding(data, cat_cols)
     X = scale_data(data, num_cols, cat_cols)
-    X_train, X_test, y_train, y_test = train_test_split(X, data[target_col], test_size=0.2)
+    X_train, X_test, y_train, y_test = train_test_split(
+        X, data[target_col], test_size=0.2
+    )
 
     mlr_model, sgd_model = fit_models(X_train, y_train)
     print_result(X_test, y_test, [mlr_model, sgd_model])
--- some_code_2.py	2023-12-04 11:19:13.085849+00:00
+++ some_code_2.py	2023-12-04 11:35:18.912557+00:00
@@ -6,11 +6,11 @@
         self._w = None
         self._X_train = None
         self._y_train = None
 
     def __str__(self):
-        return f'MyLinearRegression'
+        return f"MyLinearRegression"
 
     def _get_loss(self, X: np.array, y: np.array):
         y_pred = np.dot(X, self._w)
         return -2 * np.dot(X.T, (y - y_pred)) / X.size
 
@@ -24,17 +24,24 @@
         for _ in range(epoch):
             X_i, y_i = self._get_points_by_batch(batch)
             step = self._get_loss(X_i, y_i)
             self._w = self._w - lr * step
 
-    def fit(self, X_train, y_train, w: np.array = None, batch: int = 100, lr: float = 0.9,
-            epoch: int = 1000):
+    def fit(
+        self,
+        X_train,
+        y_train,
+        w: np.array = None,
+        batch: int = 100,
+        lr: float = 0.9,
+        epoch: int = 1000,
+    ):
         self._X_train, self._y_train = np.array(X_train), np.array(y_train)
         if w is None:
             self._w = np.zeros(X_train.shape[1])
         if batch is None:
             batch = int(X_train.size * 0.3)
 
         self._stochastic_gd(batch, lr, epoch)
 
     def predict(self, X: np.array):
-        return np.dot(X, self._w)
\ No newline at end of file
+        return np.dot(X, self._w)
--- /Users/benomen/PycharmProjects/engineering_practices_hw1/some_code_1.py:before	2023-12-04 14:35:08.873550
+++ /Users/benomen/PycharmProjects/engineering_practices_hw1/some_code_1.py:after	2023-12-04 14:36:33.293311
@@ -1,11 +1,12 @@
 from enum import Enum
+
+import numpy as np
 import pandas as pd
-import numpy as np
+from sklearn.linear_model import LinearRegression, SGDRegressor
+from sklearn.metrics import (mean_absolute_percentage_error,
+                             mean_squared_error, r2_score)
+from sklearn.model_selection import train_test_split
 from sklearn.preprocessing import StandardScaler
-from sklearn.model_selection import train_test_split
-from sklearn.linear_model import LinearRegression, SGDRegressor
-from sklearn.metrics import mean_squared_error, r2_score
-from sklearn.metrics import mean_absolute_percentage_error
 
 from some_code_2 import MyLinearRegression
 
