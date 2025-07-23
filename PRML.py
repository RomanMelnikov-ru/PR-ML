import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import make_pipeline
import joblib
from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Dense
import io
import os
from itertools import product
from tqdm import tqdm
import tensorflow as tf
tf.keras.utils.set_random_seed(42)  # Фиксирует все случайные сиды
tf.config.experimental.enable_op_determinism()  # Устраняет недетерминизм GPU

# Функция для загрузки данных
def load_data():
    st.subheader("Информация о файле")
    st.write("""
    Пожалуйста, загрузите файл в формате Excel (.xlsx). 
    Файл должен содержать следующие столбцы:
    - **A, B, C, D**: Признаки (независимые переменные).
    - **E**: Целевая переменная.
    """)

    uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            st.success("Данные успешно загружены!")
            return df
        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
    return None


# Функция для расчета выборочного корреляционного отношения
def calculate_correlation_ratio(df, target_col):
    categories = df.drop(target_col, axis=1).columns
    ratios = {}
    for cat in categories:
        grouped = df.groupby(cat)[target_col]
        n = len(df)
        y_mean = df[target_col].mean()
        ss_total = ((df[target_col] - y_mean) ** 2).sum()
        ss_between = grouped.apply(lambda x: len(x) * (x.mean() - y_mean) ** 2).sum()
        eta_squared = ss_between / ss_total
        ratios[cat] = eta_squared
    return ratios


# Функция для отображения тепловых карт корреляций
def show_correlation_heatmaps(df):
    with st.expander("Тепловые карты корреляций"):
        st.subheader("Тепловая карта корреляции Пирсона")
        pearson_corr_matrix = df.corr(method="pearson").round(3)
        fig_pearson = px.imshow(
            pearson_corr_matrix,
            text_auto=True,
            color_continuous_scale="Viridis",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_pearson)

        st.subheader("Тепловая карта корреляции Спирмана")
        spearman_corr_matrix = df.corr(method="spearman").round(3)
        fig_spearman = px.imshow(
            spearman_corr_matrix,
            text_auto=True,
            color_continuous_scale="Plasma",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_spearman)

        st.subheader("Корреляционное отношение (η²) для целевой переменной E")
        try:
            eta_squared = calculate_correlation_ratio(df, "E")
            eta_df = pd.DataFrame.from_dict(eta_squared, orient='index', columns=['η²']).round(3)
            fig_eta = px.bar(
                eta_df,
                y='η²',
                labels={'index': 'Факторы', 'y': 'Корреляционное отношение η²'},
                text='η²'
            )
            fig_eta.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig_eta)

            st.subheader("Тепловая карта корреляционных отношений")
            all_vars = df.columns
            eta_matrix = pd.DataFrame(index=all_vars, columns=all_vars)

            for var1 in all_vars:
                for var2 in all_vars:
                    if var1 == var2:
                        eta_matrix.loc[var1, var2] = 1.0
                    else:
                        grouped = df.groupby(var1)[var2]
                        n = len(df)
                        y_mean = df[var2].mean()
                        ss_total = ((df[var2] - y_mean) ** 2).sum()
                        ss_between = grouped.apply(lambda x: len(x) * (x.mean() - y_mean) ** 2).sum()
                        eta_squared = ss_between / ss_total
                        eta_matrix.loc[var1, var2] = eta_squared

            eta_matrix = eta_matrix.astype(float).round(3)
            fig_eta_matrix = px.imshow(
                eta_matrix,
                text_auto=True,
                color_continuous_scale="Viridis",
                labels=dict(x="Переменная 1", y="Переменная 2", color="η²"),
                zmin=0,
                zmax=1
            )
            st.plotly_chart(fig_eta_matrix)

        except Exception as e:
            st.error(f"Ошибка при расчете корреляционного отношения: {e}")


# Функция для отображения формулы с коэффициентами
def show_formula(coefficients, intercept, feature_names, regression_type):
    if regression_type == "Линейная":
        formula = f"E = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*{feature_names[i]}"
    elif regression_type in ["Квадратическая", "Кубическая"]:
        formula = f"E = {intercept:.7f}"
        for coef, name in zip(coefficients, feature_names):
            formula += f" + {coef:.7f}*{name}"
    elif regression_type == "Логарифмическая":
        formula = f"E = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*log({feature_names[i]})"
    elif regression_type == "Lasso":
        formula = f"E = {intercept:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" + {coef:.7f}*{feature_names[i]}"
    elif regression_type == "Экспоненциальная":
        a = np.exp(intercept)
        formula = f"E = {a:.7f} * exp("
        for i, coef in enumerate(coefficients):
            formula += f"{coef:.7f}*{feature_names[i]} + "
        formula = formula.rstrip(" + ") + ")"
    elif regression_type == "Степенная":
        a = np.exp(intercept)
        formula = f"E = {a:.7f}"
        for i, coef in enumerate(coefficients):
            formula += f" * {feature_names[i]}^{coef:.7f}"
    st.subheader("Формула модели")
    st.write(formula)


# Функция для отображения графика значимости факторов
def show_feature_importance(coefficients, feature_names):
    st.subheader("График значимости факторов")
    importance = np.abs(coefficients)
    fig = px.bar(x=feature_names, y=importance, labels={"x": "Факторы", "y": "Важность"})
    st.plotly_chart(fig)


# Функция для выполнения регрессии
def run_regression(df, regression_type):
    try:
        if df.isnull().any().any():
            raise ValueError(
                "В данных есть пропущенные значения (NaN). Пожалуйста, заполните их перед запуском модели.")

        X = df[["A", "B", "C", "D"]]
        y = df["E"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if regression_type == "Линейная":
            model = make_pipeline(StandardScaler(), LinearRegression())

        elif regression_type == "Квадратическая":
            model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression())

        elif regression_type == "Кубическая":
            model = make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression())

        elif regression_type == "Логарифмическая":
            if (X_train.values <= 0).any() or (X_test.values <= 0).any():
                st.error("Логарифмическая регрессия требует положительных значений в данных.")
                return
            log_x_train = np.log(X_train)
            log_x_test = np.log(X_test)
            model = make_pipeline(StandardScaler(), LinearRegression())
            model.fit(log_x_train, y_train)
            y_pred = model.predict(log_x_test)

        elif regression_type == "Экспоненциальная":
            if (y_train <= 0).any() or (y_test <= 0).any():
                st.error("Экспоненциальная регрессия требует положительных значений в целевой переменной.")
                return
            y_train_log = np.log(y_train)
            y_test_log = np.log(y_test)
            model = LinearRegression()
            model.fit(X_train, y_train_log)
            y_pred_log = model.predict(X_test)
            y_pred = np.exp(y_pred_log)
            coefficients = model.coef_
            intercept = model.intercept_
            feature_names = X.columns
            show_formula(coefficients, intercept, feature_names, regression_type)

        elif regression_type == "Степенная":
            if (X_train.values <= 0).any() or (y_train <= 0).any():
                st.error("Степенная регрессия требует положительных значений в данных.")
                return
            X_train_log = np.log(X_train)
            X_test_log = np.log(X_test)
            y_train_log = np.log(y_train)
            y_test_log = np.log(y_test)
            model = LinearRegression()
            model.fit(X_train_log, y_train_log)
            y_pred_log = model.predict(X_test_log)
            y_pred = np.exp(y_pred_log)
            coefficients = model.coef_
            intercept = model.intercept_
            feature_names = X.columns
            show_formula(coefficients, intercept, feature_names, regression_type)

        elif regression_type == "Lasso":
            model = make_pipeline(StandardScaler(), Lasso(alpha=0.1))

        elif regression_type == "SVR (Метод опорных векторов)":
            model = make_pipeline(StandardScaler(), SVR(kernel='rbf'))
            st.write(f"Используемое ядро: {model.named_steps['svr'].kernel}")

        elif regression_type == "Decision Tree (Решающее дерево)":
            model = DecisionTreeRegressor(random_state=42)
            st.write(
                "Максимальная глубина дерева: None (дерево растёт до тех пор, пока все листья не станут чистыми или пока не будет достигнуто минимальное количество образцов в листе).")

        elif regression_type == "Random Forest (Случайный лес)":
            model = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)
            st.write(f"Количество деревьев: 100")
            st.write(
                f"Максимальная глубина деревьев: None (дерево растёт до тех пор, пока все листья не станут чистыми).")

        elif regression_type == "Gradient Boosting (Градиентный бустинг)":
            model = GradientBoostingRegressor(random_state=42)
            st.write("Количество деревьев: 100 (по умолчанию)")
            st.write("Максимальная глубина деревьев: 3 (по умолчанию)")

        elif regression_type == "Gaussian Processes (Гауссовские процессы)":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            model = GaussianProcessRegressor(kernel=kernel, random_state=42)
            st.write(f"Используемое ядро: {kernel}")

        elif regression_type == "Neural Network (Нейронная сеть)":
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='glorot_uniform'))  # Явная инициализация
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            history = model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0, validation_split=0.2)
            y_pred = model.predict(X_test).flatten()
            st.subheader("График обучения")
            fig = px.line(history.history, y=['loss', 'val_loss'], labels={"value": "Loss", "index": "Epoch"})
            st.plotly_chart(fig)
            st.write(
                "Архитектура сети: 64 нейрона (входной слой), 32 нейрона (скрытый слой), 1 нейрон (выходной слой).")

        if regression_type not in ["Экспоненциальная", "Степенная", "Neural Network (Нейронная сеть)"]:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        st.subheader("Метрики модели")
        st.write(f"MSE (Среднеквадратичная ошибка): {mse:.2f}")
        st.write(f"RMSE (Корень из среднеквадратичной ошибки): {rmse:.2f}")
        st.write(f"MAE (Средняя абсолютная ошибка): {mae:.2f}")
        st.write(f"MAPE (Средняя абсолютная процентная ошибка): {mape:.2f}%")
        st.write(f"R² (Коэффициент детерминации): {r2:.2f}")

        if regression_type in ["Линейная", "Квадратическая", "Кубическая", "Логарифмическая", "Lasso",
                               "Экспоненциальная", "Степенная"]:
            if regression_type in ["Линейная", "Квадратическая", "Кубическая", "Логарифмическая", "Lasso"]:
                coefficients = model.named_steps["linearregression"].coef_ if regression_type != "Lasso" else \
                model.named_steps["lasso"].coef_
                intercept = model.named_steps["linearregression"].intercept_ if regression_type != "Lasso" else \
                model.named_steps["lasso"].intercept_
                feature_names = (
                    X.columns
                    if regression_type == "Линейная"
                    else model.named_steps["polynomialfeatures"].get_feature_names_out(X.columns)
                    if regression_type in ["Квадратическая", "Кубическая"]
                    else X.columns
                )
            show_formula(coefficients, intercept, feature_names, regression_type)
            if regression_type != "Экспоненциальная" and regression_type != "Степенная":
                show_feature_importance(coefficients, feature_names)

        st.subheader("График фактических vs предсказанных значений")
        fig1 = px.scatter(x=y_test, y=y_pred, labels={"x": "Фактические значения", "y": "Предсказанные значения"})
        fig1.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], mode="lines",
                                  name="Идеальная линия"))
        st.plotly_chart(fig1)

        st.subheader("График остатков")
        residuals = y_test - y_pred
        fig2 = px.scatter(x=y_pred, y=residuals, labels={"x": "Предсказанные значения", "y": "Остатки"})
        fig2.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig2)

        if regression_type in [
            "SVR (Метод опорных векторов)", "Decision Tree (Решающее дерево)", "Random Forest (Случайный лес)",
            "Gradient Boosting (Градиентный бустинг)", "Gaussian Processes (Гауссовские процессы)",
            "Neural Network (Нейронная сеть)"
        ]:
            save_model_to_file(model, regression_type)

    except ValueError as e:
        st.error(f"Ошибка: {e}")
    except Exception as e:
        st.error(f"Ошибка при выполнении регрессии: {e}")


# Функция для сохранения модели
def save_model_to_file(model, regression_type):
    if model is None:
        st.error("Модель не обучена. Сначала обучите модель.")
        return

    try:
        if regression_type == "Neural Network (Нейронная сеть)":
            file_format = st.selectbox("Выберите формат файла", [".h5"])
        else:
            file_format = st.selectbox("Выберите формат файла", [".pkl", ".joblib"])

        default_filename = f"model{file_format}"
        file_path = st.text_input("Введите имя файла для сохранения модели", value=default_filename)

        if st.button("Сохранить модель"):
            if regression_type == "Neural Network (Нейронная сеть)":
                save_model(model, file_path)
            else:
                if file_format == ".pkl":
                    joblib.dump(model, file_path)
                elif file_format == ".joblib":
                    joblib.dump(model, file_path)

            with open(file_path, "rb") as f:
                st.download_button(
                    label="Скачать модель",
                    data=f,
                    file_name=file_path,
                    mime="application/octet-stream"
                )
            st.success(f"Модель сохранена в файл: {file_path}")
            os.remove(file_path)

    except Exception as e:
        st.error(f"Ошибка при сохранении модели: {e}")


# Функция для сравнения моделей
def compare_models(df):
    st.subheader("Сравнение моделей по коэффициенту детерминации R²")

    if df is None:
        st.warning("Загрузите данные для сравнения моделей")
        return

    try:
        if df.isnull().any().any():
            raise ValueError(
                "В данных есть пропущенные значения (NaN). Пожалуйста, заполните их перед сравнением моделей.")

        X = df[["A", "B", "C", "D"]]
        y = df["E"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        models = {
            "Линейная": make_pipeline(StandardScaler(), LinearRegression()),
            "Квадратическая": make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), LinearRegression()),
            "Кубическая": make_pipeline(StandardScaler(), PolynomialFeatures(degree=3), LinearRegression()),
            "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
            "SVR": make_pipeline(StandardScaler(), SVR(kernel='rbf')),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "Gaussian Processes": GaussianProcessRegressor(kernel=C(1.0) * RBF(10), random_state=42)
        }

        special_models = ["Логарифмическая", "Экспоненциальная", "Степенная"]

        results = []

        # Оцениваем стандартные модели
        for name, model in tqdm(models.items(), desc="Оценка моделей"):
            try:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                r2 = r2_score(y_test, y_pred)
                results.append({"Модель": name, "R²": r2, "Тип": "standard"})
            except Exception as e:
                results.append({"Модель": name, "R²": f"Ошибка: {str(e)}", "Тип": "error"})

        # Оцениваем специальные модели
        for model_type in special_models:
            try:
                if model_type == "Логарифмическая":
                    if (X_train.values <= 0).any() or (X_test.values <= 0).any():
                        results.append({"Модель": model_type, "R²": "Требуются положительные значения", "Тип": "error"})
                        continue
                    log_x_train = np.log(X_train)
                    log_x_test = np.log(X_test)
                    model = LinearRegression()
                    model.fit(log_x_train, y_train)
                    y_pred = model.predict(log_x_test)

                elif model_type == "Экспоненциальная":
                    if (y_train <= 0).any() or (y_test <= 0).any():
                        results.append({"Модель": model_type, "R²": "Требуются положительные значения", "Тип": "error"})
                        continue
                    y_train_log = np.log(y_train)
                    model = LinearRegression()
                    model.fit(X_train, y_train_log)
                    y_pred_log = model.predict(X_test)
                    y_pred = np.exp(y_pred_log)

                elif model_type == "Степенная":
                    if (X_train.values <= 0).any() or (y_train <= 0).any():
                        results.append({"Модель": model_type, "R²": "Требуются положительные значения", "Тип": "error"})
                        continue
                    X_train_log = np.log(X_train)
                    y_train_log = np.log(y_train)
                    model = LinearRegression()
                    model.fit(X_train_log, y_train_log)
                    y_pred_log = model.predict(np.log(X_test))
                    y_pred = np.exp(y_pred_log)

                r2 = r2_score(y_test, y_pred)
                results.append({"Модель": model_type, "R²": r2, "Тип": "standard"})
            except Exception as e:
                results.append({"Модель": model_type, "R²": f"Ошибка: {str(e)}", "Тип": "error"})

        # Оцениваем нейронную сеть
        try:
            model = Sequential()
            model.add(Dense(64, input_dim=X_train.shape[1], kernel_initializer='glorot_uniform'))  # Явная инициализация
            model.add(Dense(32, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)
            y_pred = model.predict(X_test).flatten()
            r2 = r2_score(y_test, y_pred)
            results.append({"Модель": "Нейронная сеть", "R²": r2, "Тип": "standard"})
        except Exception as e:
            results.append({"Модель": "Нейронная сеть", "R²": f"Ошибка: {str(e)}", "Тип": "error"})

        # Создаем DataFrame и сортируем
        results_df = pd.DataFrame(results)

        # Разделяем на успешные модели и ошибки
        success_df = results_df[results_df["Тип"] == "standard"].copy()
        error_df = results_df[results_df["Тип"] == "error"].copy()

        # Сортируем успешные модели по R²
        success_df = success_df.sort_values(by="R²", ascending=False)

        # Объединяем результаты
        final_df = pd.concat([success_df, error_df]).drop(columns=["Тип"])

        # Форматируем вывод
        def format_r2(val):
            if isinstance(val, str):
                return val
            return f"{val:.4f}"

        final_df["R²"] = final_df["R²"].apply(format_r2)

        # Отображаем результаты
        st.dataframe(
            final_df.style
            .apply(lambda x: ["background: lightgreen" if isinstance(x["R²"], float) and x["R²"] > 0.7
                              else "background: lightyellow" if isinstance(x["R²"], float) and x["R²"] > 0.5
            else "background: lightcoral" if isinstance(x["R²"], float)
            else "" for i in range(len(x))], axis=1)
            .set_properties(**{'text-align': 'center'})
        )

        st.markdown("""
        **Пояснения:**
        - <span style="background-color:lightgreen">Зеленый</span>: Отличная модель (R² > 0.7)
        - <span style="background-color:lightyellow">Желтый</span>: Хорошая модель (R² > 0.5)
        - <span style="background-color:lightcoral">Красный</span>: Плохая модель (R² ≤ 0.5)
        - Ошибки выделены стандартным цветом
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Ошибка при сравнении моделей: {e}")


# Основной интерфейс
st.title("Полиномиальная регрессия и машинное обучение")

# Загрузка данных
df = load_data()
if df is not None:
    show_correlation_heatmaps(df)

    if st.button("Сравнить все модели по R²"):
        compare_models(df)

    regression_types = [
        "Линейная", "Квадратическая", "Кубическая", "Логарифмическая", "Экспоненциальная", "Степенная",
        "Lasso", "SVR (Метод опорных векторов)", "Decision Tree (Решающее дерево)", "Random Forest (Случайный лес)",
        "Gradient Boosting (Градиентный бустинг)", "Gaussian Processes (Гауссовские процессы)",
        "Neural Network (Нейронная сеть)"
    ]
    regression_type = st.selectbox("Выберите тип регрессии", regression_types)

    if st.button("Запустить регрессию"):
        run_regression(df, regression_type)
