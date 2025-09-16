import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import uuid
import hashlib
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import os
from scipy import stats
from scipy.stats import pearsonr, spearmanr, shapiro
import tensorflow as tf
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
import warnings
import xgboost as xgb

# Настройка логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Подавление предупреждений
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')
# Воспроизводимость
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# --- Попытка импорта XGBoost ---
try:
    from xgboost import XGBRegressor

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    st.warning("XGBoost не установлен. Для установки: `pip install xgboost`")


# Кэширование
@st.cache_resource(show_spinner=False)
def load_data_cached(uploaded_file):
    return pd.read_excel(uploaded_file)


@st.cache_data(show_spinner=False)
def calculate_correlation_matrix(df, method='pearson'):
    return df.corr(method=method)


@st.cache_data(show_spinner=False)
def calculate_vif_cached(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data["VIF"] = vif_data["VIF"].round(2)
    return vif_data


def load_data():
    st.subheader("Загрузка данных")
    st.write("""
    Пожалуйста, загрузите файл в формате Excel (.xlsx). 
    Файл должен содержать следующие столбцы:
    - **B, C, D, E, F, G, H**: Признаки (независимые переменные).
    - **A**: Целевая переменная.
    """)

    uploaded_file = st.file_uploader("Загрузите файл Excel", type=["xlsx"])

    if uploaded_file is not None:
        try:
            with st.spinner("Загрузка данных..."):
                df = load_data_cached(uploaded_file)

            # Удаление полностью пустых столбцов
            df = df.dropna(axis=1, how='all')

            # Проверка на пропуски и бесконечности
            has_missing = df.isnull().any().any()
            has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()

            if has_missing or has_inf:
                st.warning("Обнаружены пропущенные значения или бесконечности. Выберите метод обработки:")
                # Выбор стратегии обработки
                col1, col2 = st.columns(2)
                with col1:
                    strategy = st.radio(
                        "Метод обработки пропусков:",
                        ["Медиана", "Среднее", "Константа", "Удалить строки"],
                        index=3,
                        key="missing_values_strategy"
                    )
                with col2:
                    constant_value = None
                    if strategy == "Константа":
                        constant_value = st.number_input("Значение для замены", value=0.0, key="fill_constant_value")

                # Применение выбранного метода
                df = handle_missing_values(df, strategy, constant_value)

                st.success(f"Пропуски обработаны методом: {strategy}")

            # Проверка обязательных столбцов
            if 'A' not in df.columns:
                st.error("Ошибка: В данных отсутствует столбец 'A' (целевая переменная)")
                return None

            feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]
            if not feature_cols:
                st.error("Ошибка: В данных отсутствуют столбцы с признаками (B, C, D, E, F, G или H)")
                return None

            st.success(f"Данные успешно загружены! Используются признаки: {', '.join(feature_cols)}")
            return df

        except Exception as e:
            st.error(f"Ошибка при загрузке файла: {e}")
            return None
    return None


def get_scaler_from_name(scaling_method):
    if scaling_method == "StandardScaler (стандартизация)":
        return StandardScaler()
    elif scaling_method == "MinMaxScaler (нормализация)":
        return MinMaxScaler()
    elif scaling_method == "RobustScaler (устойчивый)":
        return RobustScaler()
    else:
        return None


def handle_missing_values(df, strategy="Медиана", constant_value=None):
    """Обработка пропущенных значений с выбором стратегии."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Замена бесконечностей на NaN
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Выбор стратегии
    if strategy == "Удалить строки":
        df = df.dropna(subset=numeric_cols)
    else:
        if strategy == "Медиана":
            imputer = SimpleImputer(strategy='median')
        elif strategy == "Среднее":
            imputer = SimpleImputer(strategy='mean')
        elif strategy == "Константа":
            imputer = SimpleImputer(strategy='constant', fill_value=constant_value)

        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

    # Удаление столбцов с нулевой дисперсией (опционально)
    selector = VarianceThreshold()
    selector.fit(df[numeric_cols])
    df = df.iloc[:, selector.get_support(indices=True)]

    return df


def handle_outliers_iqr(df, columns=None):
    df_clean = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if columns is not None:
        numeric_cols = [c for c in numeric_cols if c in columns]
    total_outliers = 0
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        total_outliers += len(outliers)
        df_clean.loc[outliers, col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    if total_outliers > 0:
        st.info(f"Обработка выбросов (IQR): заменено {total_outliers} значений")
    else:
        st.info("Обработка выбросов (IQR): выбросы не обнаружены")
    return df_clean


def calculate_vif(X):
    try:
        # Проверяем, что X не пустой и содержит числовые данные
        if X.empty or not all(X.dtypes.apply(lambda x: np.issubdtype(x, np.number))):
            return pd.DataFrame({"feature": X.columns, "VIF": [np.nan] * len(X.columns)})

        # Удаляем строки с пропущенными значениями для расчета VIF
        X_clean = X.dropna()
        if len(X_clean) < 2:
            return pd.DataFrame({"feature": X.columns, "VIF": [np.nan] * len(X.columns)})

        return calculate_vif_cached(X_clean)
    except Exception as e:
        logger.error(f"Ошибка при расчете VIF: {str(e)}")
        return pd.DataFrame({"feature": X.columns, "VIF": [np.nan] * len(X.columns)})


def show_descriptive_analysis(df):
    st.subheader("Описательная статистика")
    desc = df.describe().T
    desc = desc.rename(columns={
        'count': 'Количество',
        'mean': 'Среднее',
        'std': 'Стандартное отклонение',
        'min': 'Минимум',
        '25%': '25-й перцентиль (Q1)',
        '50%': 'Медиана (Q2)',
        '75%': '75-й перцентиль (Q3)',
        'max': 'Максимум'
    })
    desc.index.name = 'Признак'
    st.dataframe(desc.style.format(precision=4))

    st.subheader("Распределение переменных")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f'Распределение признака {col}',
            labels={'x': col, 'y': 'Частота'}
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Парные распределения")
    fig = px.scatter_matrix(df[numeric_cols], dimensions=numeric_cols)
    st.plotly_chart(fig, use_container_width=True)


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


def calculate_pvalues(df, method='pearson'):
    df = df._get_numeric_data()
    cols = df.columns
    n = len(cols)
    p = pd.DataFrame(np.zeros((n, n)), columns=cols, index=cols)
    for i, r in enumerate(cols):
        for j, c in enumerate(cols):
            if i == j:
                p.iloc[i, j] = 0
                continue
            x = df[r]
            y = df[c]
            if x.nunique() == 1 or y.nunique() == 1:
                p.iloc[i, j] = np.nan
                continue
            try:
                if method == 'pearson':
                    p.iloc[i, j] = pearsonr(x, y)[1]
                elif method == 'spearman':
                    p.iloc[i, j] = spearmanr(x, y)[1]
            except:
                p.iloc[i, j] = np.nan
    return p


def add_significance_stars(corr_matrix, p_matrix):
    sig_matrix = np.empty_like(corr_matrix, dtype=object)
    for i in range(corr_matrix.shape[0]):
        for j in range(corr_matrix.shape[1]):
            if i == j:
                sig_matrix[i, j] = ""
                continue
            corr = corr_matrix.iloc[i, j]
            p = p_matrix.iloc[i, j]
            if pd.isna(p) or pd.isna(corr):
                sig_matrix[i, j] = "NaN"
                continue
            if p < 0.001:
                sig = f"{corr:.2f}***"
            elif p < 0.01:
                sig = f"{corr:.2f}**"
            elif p < 0.05:
                sig = f"{corr:.2f}*"
            else:
                sig = f"{corr:.2f}"
            sig_matrix[i, j] = sig
    return sig_matrix


def show_model_significance(X, y, model):
    try:
        X_with_const = sm.add_constant(X)
        model_sm = sm.OLS(y, X_with_const).fit()
        st.subheader("Статистическая значимость модели (ANOVA)")
        st.write(model_sm.summary())
        p_value = model_sm.f_pvalue
        st.write(f"F-статистика: {model_sm.fvalue:.4f}, p-value: {p_value:.4f}")
        st.markdown("""
        📊 F-статистика:
        - Проверяет значимость модели в целом
        - Нулевая гипотеза: все коэффициенты = 0 (модель бесполезна)
        - Альтернатива: хотя бы один коэффициент ≠ 0
        - Интерпретация:
        * Большое F и малое p-value (<0.05) - модель значима
        * F = (Объясненная дисперсия) / (Необъясненная дисперсия)
        - Чем больше F, тем лучше модель объясняет данные
        """)
        if p_value < 0.05:
            st.success("Модель статистически значима (p < 0.05)")
        else:
            st.warning("Модель не является статистически значимой (p ≥ 0.05)")
        return model_sm.pvalues[1:].tolist()
    except Exception as e:
        st.warning(f"Не удалось проверить значимость модели: {str(e)}")
        return None


def predict_with_model(model, model_name, X_input, result):
    X_input = X_input.copy()
    scaling_method = result.get("scaling_method", "Нет")
    scaler = result.get("scaler", None)

    try:
        if model_name != "Нейронная сеть" and scaling_method != "Нет" and scaler is not None:
            X_scaled = scaler.transform(X_input)
            X_input = pd.DataFrame(X_scaled, columns=X_input.columns)

        if model_name == "Нейронная сеть":
            pred = model.predict(X_input).flatten()[0]
        elif model_name == "Логарифмическая":
            X_log = np.log(X_input.clip(lower=1e-9))
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)
            pred = model.predict(X_log)[0]
        elif model_name == "Экспоненциальная":
            pred = np.exp(model.predict(X_input)[0])
        elif model_name == "Степенная":
            X_log = np.log(X_input.clip(lower=1e-9))
            X_log = np.nan_to_num(X_log, posinf=0, neginf=0)
            pred = np.exp(model.predict(X_log)[0])
        else:
            pred = model.predict(X_input)[0]
        return float(pred) if pred is not None else None
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return None


def show_formula(coefficients, intercept, feature_names, regression_type, p_values=None, model_pipeline=None,
                 result=None):
    scaler = None
    sigma = None
    mu = None

    if model_pipeline is not None and hasattr(model_pipeline, 'named_steps'):
        for step_name, step in model_pipeline.named_steps.items():
            if isinstance(step, (StandardScaler, MinMaxScaler, RobustScaler)):
                scaler = step
                break

    if scaler is None and result is not None:
        scaler = result.get("scaler")

    if scaler is not None:
        if isinstance(scaler, StandardScaler):
            sigma = scaler.scale_
            mu = scaler.mean_
        elif isinstance(scaler, MinMaxScaler):
            sigma = scaler.data_max_ - scaler.data_min_
            mu = scaler.data_min_
        elif isinstance(scaler, RobustScaler):
            sigma = scaler.scale_
            mu = scaler.center_

    original_coefs = coefficients.copy()
    original_intercept = intercept

    if sigma is not None and mu is not None:
        try:
            if regression_type in ["Линейная", "Квадратическая", "Кубическая", "Lasso"]:
                original_coefs = coefficients / sigma
                original_intercept = intercept - np.sum(coefficients * mu / sigma)
            elif regression_type == "Логарифмическая":
                original_coefs = coefficients / sigma
                original_intercept = intercept - np.sum(coefficients * mu / sigma)
            elif regression_type == "Экспоненциальная":
                original_coefs = coefficients / sigma
                original_intercept = intercept - np.sum(coefficients * mu / sigma)
            elif regression_type == "Степенная":
                original_coefs = coefficients / sigma
                original_intercept = intercept - np.sum(coefficients * mu / sigma)
        except Exception as e:
            st.warning(f"Ошибка при пересчёте коэффициентов: {str(e)}")
            st.caption("Формула показана в масштабированных данных")

    formula_parts = [f"{original_intercept:.4f}"]

    for i, (coef, name) in enumerate(zip(original_coefs, feature_names)):
        significance = ""
        if p_values is not None and i < len(p_values):
            if p_values[i] < 0.001:
                significance = "***"
            elif p_values[i] < 0.01:
                significance = "**"
            elif p_values[i] < 0.05:
                significance = "*"

        if regression_type == "Логарифмическая":
            term = f"{coef:.4f}{significance}*log({name})"
        elif regression_type in ["Квадратическая", "Кубическая"] and "^" in name:
            base, power = name.split("^")
            term = f"{coef:.4f}{significance}*{base}^{power}"
        elif regression_type == "Экспоненциальная":
            continue
        elif regression_type == "Степенная":
            continue
        else:
            term = f"{coef:.4f}{significance}*{name}"

        formula_parts.append(term)

    if regression_type == "Логарифмическая":
        formula = "A = " + " + ".join(formula_parts)
    elif regression_type == "Экспоненциальная":
        a0 = np.exp(original_intercept)
        terms = [f"{coef:.4f}{significance}*{name}" for coef, name in zip(
            original_coefs,
            feature_names
        )]
        formula = f"A = {a0:.4f} * exp(" + " + ".join(terms) + ")"
    elif regression_type == "Степенная":
        a0 = np.exp(original_intercept)
        terms = [f"{name}^{coef:.4f}{significance}" for coef, name in zip(
            original_coefs,
            feature_names
        )]
        formula = f"A = {a0:.4f} * " + " * ".join(terms)
    else:
        formula = "A = " + " + ".join(formula_parts)

    st.subheader("Формула модели (в исходном масштабе)")
    st.write(formula)

    if p_values is not None:
        st.markdown("""
        **Обозначения уровня значимости:**
        - \*** p < 0.001 — очень высокая значимость
        - \** p < 0.01 — высокая значимость
        - \* p < 0.05 — статистически значимо
        """)


def show_feature_importance(coefficients, feature_names, p_values=None, model=None, X=None, y=None):
    if coefficients is not None:
        importance = np.abs(coefficients)
        if p_values is not None:
            significance = []
            for p in p_values:
                if p < 0.001:
                    significance.append("***")
                elif p < 0.01:
                    significance.append("**")
                elif p < 0.05:
                    significance.append("*")
                else:
                    significance.append("")
            text = [f"{imp:.4f}{sig}" for imp, sig in zip(importance, significance)]
        else:
            text = [f"{imp:.4f}" for imp in importance]
        fig = px.bar(
            x=feature_names,
            y=importance,
            text=text,
            labels={"x": "Факторы", "y": "Важность"},
            title="Важность признаков (коэффициенты модели)"
        )
        fig.update_traces(textposition='outside')
        fig_key = f"feature_importance_{uuid.uuid4()}"
        st.plotly_chart(fig, use_container_width=True, key=fig_key)
    elif model is not None and X is not None and y is not None:
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                fig = px.bar(
                    x=feature_names,
                    y=importance,
                    text=[f"{imp:.4f}" for imp in importance],
                    labels={"x": "Факторы", "y": "Важность"},
                    title="Важность признаков (встроенная важность)"
                )
                fig.update_traces(textposition='outside')
                st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{uuid.uuid4()}")
            else:
                with st.spinner("Вычисление важности признаков (permutation importance)..."):
                    if hasattr(model, 'predict'):
                        result = permutation_importance(model, X, y, n_repeats=10, random_state=42,
                                                        scoring='neg_mean_squared_error')
                        importance = result.importances_mean
                        fig = px.bar(
                            x=feature_names,
                            y=importance,
                            text=[f"{imp:.4f}" for imp in importance],
                            labels={"x": "Факторы", "y": "Важность"},
                            title="Важность признаков (permutation importance)"
                        )
                        fig.update_traces(textposition='outside')
                        st.plotly_chart(fig, use_container_width=True, key=f"feature_importance_{uuid.uuid4()}")
                    else:
                        st.warning("Не удалось вычислить важность признаков: модель не имеет метода predict")
        except Exception as e:
            st.warning(f"Не удалось вычислить важность признаков: {str(e)}")


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        fig = px.scatter(
            x=y_true,
            y=y_pred,
            labels={'x': 'Фактические значения', 'y': 'Предсказанные значения'},
            trendline='lowess',
            trendline_color_override='green',
            title=f"Фактические vs предсказанные значения ({model_name})"
        )
        min_val = min(min(y_true), min(y_pred))
        max_val = max(max(y_true), max(y_pred))
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Идеальная линия',
                                 line=dict(color='red', dash='dash')))
        st.plotly_chart(fig, use_container_width=True, key=f"actual_vs_predicted_{uuid.uuid4()}")
    except Exception as e:
        st.error(f"Ошибка при построении графика: {str(e)}")


def plot_residuals(y_true, y_pred, model_name):
    try:
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()
        residuals = y_true - y_pred
        fig = px.scatter(
            x=y_pred,
            y=residuals,
            labels={'x': 'Предсказанные значения', 'y': 'Остатки'},
            title=f"График остатков ({model_name})"
        )
        fig.add_hline(y=0, line_dash='dash', line_color='red')
        st.plotly_chart(fig, use_container_width=True, key=f"residuals_{uuid.uuid4()}")
        st.subheader("Проверка нормальности остатков")
        try:
            stat, p = shapiro(residuals)
            st.write(f"Тест Шапиро-Уилка: статистика = {stat:.4f}, p-value = {p:.4f}")
            if p > 0.05:
                st.success("Остатки распределены нормально (p > 0.05)")
            else:
                st.warning("Остатки не распределены нормально (p ≤ 0.05)")
        except Exception as e:
            st.warning(f"Не удалось выполнить тест на нормальность: {str(e)}")
    except Exception as e:
        st.error(f"Ошибка при построении графика остатков: {str(e)}")


def get_poly_features_from_pipeline(pipeline):
    for step_name, step in pipeline.named_steps.items():
        if isinstance(step, PolynomialFeatures):
            return step
    return None


def plot_response_surface(model, X, y, feature_names, regression_type, model_key=""):
    try:
        if len(feature_names) < 2:
            st.warning("Для построения поверхности отклика нужно как минимум 2 признака")
            return
        st.subheader("Поверхность отклика")
        poly_transformer = None
        original_features = feature_names.copy()
        if hasattr(model, 'named_steps'):
            poly_transformer = get_poly_features_from_pipeline(model)
            if poly_transformer is not None:
                original_features = poly_transformer.feature_names_in_

        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Ось X", original_features, index=0, key=f"x_axis_{model_key}_{uuid.uuid4()}")
        with col2:
            y_axis = st.selectbox("Ось Y", [f for f in original_features if f != x_axis], index=0,
                                  key=f"y_axis_{model_key}_{uuid.uuid4()}")

        fixed_values = {}
        for feature in original_features:
            if feature not in [x_axis, y_axis]:
                if len(X[feature].unique()) > 1:
                    key_suffix = hashlib.md5((feature + model_key).encode()).hexdigest()[:8]
                    key = f"fixed_{feature}_{model_key}_{key_suffix}"
                    fixed_values[feature] = st.slider(
                        f"Значение для '{feature}'",
                        min_value=float(X[feature].min()),
                        max_value=float(X[feature].max()),
                        value=float(X[feature].median()),
                        key=key
                    )
                else:
                    fixed_values[feature] = X[feature].iloc[0]

        X_filtered = X.copy()
        y_filtered = y.copy()

        for feature, fixed_val in fixed_values.items():
            if feature in X_filtered.columns:
                min_val = X_filtered[feature].min()
                max_val = X_filtered[feature].max()
                range_val = max_val - min_val
                tol = 0.1 * range_val if range_val != 0 else 0.1
                mask = (X_filtered[feature] >= fixed_val - tol) & (X_filtered[feature] <= fixed_val + tol)
                X_filtered = X_filtered[mask]
                y_filtered = y_filtered[mask]

        if len(X_filtered) == 0:
            st.info("Нет данных, близких к выбранным фиксированным значениям. Точки не отображаются.")
        else:
            st.info(f"На графике отображаются **{len(X_filtered)}** точек, близких к фиксированным значениям.")

        x_range = np.linspace(X[x_axis].min(), X[x_axis].max(), 20)
        y_range = np.linspace(X[y_axis].min(), X[y_axis].max(), 20)
        xx, yy = np.meshgrid(x_range, y_range)

        predict_data = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
        for feature, value in fixed_values.items():
            predict_data[feature] = value
        predict_data = predict_data[original_features]

        if regression_type == "Логарифмическая":
            predict_data = np.log(predict_data.clip(lower=1e-9))
            predict_data = np.nan_to_num(predict_data, posinf=0, neginf=0)
        elif regression_type == "Степенная":
            predict_data = np.log(predict_data.clip(lower=1e-9))
            predict_data = np.nan_to_num(predict_data, posinf=0, neginf=0)

        try:
            if regression_type == "Экспоненциальная":
                zz = np.exp(model.predict(predict_data)).reshape(xx.shape)
            elif regression_type == "Степенная":
                zz = np.exp(model.predict(predict_data)).reshape(xx.shape)
            elif regression_type == "Логарифмическая":
                zz = model.predict(predict_data).reshape(xx.shape)
            else:
                zz = model.predict(predict_data).reshape(xx.shape)

            fig = go.Figure()

            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale='Viridis',
                opacity=0.8,
                name='Поверхность отклика',
                showscale=True,
                colorbar=dict(title="Предсказание")
            ))

            if len(X_filtered) > 0:
                fig.add_trace(go.Scatter3d(
                    x=X_filtered[x_axis],
                    y=X_filtered[y_axis],
                    z=y_filtered,
                    mode='markers',
                    marker=dict(size=2, color='red', opacity=0.8),
                    name='Исходные данные (близкие)',
                    showlegend=True
                ))

            fig.update_layout(
                title=f'Поверхность отклика: {x_axis} vs {y_axis} ({regression_type})',
                scene=dict(
                    xaxis_title=x_axis,
                    yaxis_title=y_axis,
                    zaxis_title='Предсказание'
                ),
                width=900,
                height=700,
                margin=dict(l=0, r=0, b=0, t=80),
                legend=dict(
                    x=0.7,
                    y=0.9,
                    bgcolor='rgba(255,255,255,0.8)',
                    bordercolor='gray',
                    borderwidth=1
                )
            )

            st.plotly_chart(fig, use_container_width=True, key=f"response_surface_{uuid.uuid4()}")

        except Exception as e:
            st.error(f"Ошибка предсказания: {str(e)}")

    except Exception as e:
        st.error(f"Ошибка построения поверхности: {str(e)}")


def save_trained_model(model, model_name):
    try:
        if isinstance(model, Sequential):
            buffer = BytesIO()
            # Сохраняем модель во временный .h5 файл
            temp_h5 = f"{model_name}_model.h5"
            model.save(temp_h5)  # Keras 3: формат определяется по расширению
            # Читаем в буфер
            with open(temp_h5, "rb") as f:
                buffer.write(f.read())
            buffer.seek(0)
            os.remove(temp_h5)  # Удаляем временный файл

            st.download_button(
                label=f"Скачать модель {model_name}",
                data=buffer,
                file_name=f"{model_name}_model.h5",
                mime="application/octet-stream",
                key=f"download_{model_name.replace(' ', '_')}_{uuid.uuid4()}"
            )
        else:
            # Для остальных моделей (sklearn) — как было
            buffer = BytesIO()
            joblib.dump(model, buffer)
            buffer.seek(0)
            st.download_button(
                label=f"Скачать модель {model_name}",
                data=buffer,
                file_name=f"{model_name}_model.pkl",
                mime="application/octet-stream",
                key=f"download_{model_name.replace(' ', '_')}_{uuid.uuid4()}"
            )
    except Exception as e:
        st.error(f"Ошибка при сохранении модели {model_name}: {str(e)}")


def train_model(reg_type, X_train, y_train, X_test, y_test, feature_cols, positive_mask_train, positive_mask_test,
                scaling_method):
    try:
        model = None
        coefficients = None
        intercept = None
        feature_names = feature_cols.copy()
        original_features = feature_cols.copy()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        if scaling_method == "StandardScaler (стандартизация)":
            scaler = StandardScaler()
        elif scaling_method == "MinMaxScaler (нормализация)":
            scaler = MinMaxScaler()
        elif scaling_method == "RobustScaler (устойчивый)":
            scaler = RobustScaler()
        else:
            scaler = None

        if reg_type == "Линейная":
            steps = [LinearRegression()]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_

        elif reg_type == "Квадратическая":
            steps = [PolynomialFeatures(degree=2, include_bias=False)]
            if scaler is not None:
                steps.append(scaler)
            steps.append(LinearRegression())
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_
            poly = model.named_steps['polynomialfeatures']
            feature_names = poly.get_feature_names_out(X_train.columns)

        elif reg_type == "Кубическая":
            steps = [PolynomialFeatures(degree=3, include_bias=False)]
            if scaler is not None:
                steps.append(scaler)
            steps.append(LinearRegression())
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_
            poly = model.named_steps['polynomialfeatures']
            feature_names = poly.get_feature_names_out(X_train.columns)

        elif reg_type == "Логарифмическая":
            steps = [LinearRegression()]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            X_log_train = np.log(X_train[positive_mask_train].clip(lower=1e-9))
            X_log_train = np.nan_to_num(X_log_train, posinf=0, neginf=0)
            y_train_pos = y_train[positive_mask_train]
            model.fit(X_log_train, y_train_pos)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_

        elif reg_type == "Экспоненциальная":
            steps = [LinearRegression()]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            y_log_train = np.log(y_train[positive_mask_train].clip(lower=1e-9))
            y_log_train = np.nan_to_num(y_log_train, posinf=0, neginf=0)
            X_train_pos = X_train[positive_mask_train]
            model.fit(X_train_pos, y_log_train)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_

        elif reg_type == "Степенная":
            steps = [LinearRegression()]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            X_log_train = np.log(X_train[positive_mask_train].clip(lower=1e-9))
            X_log_train = np.nan_to_num(X_log_train, posinf=0, neginf=0)
            y_log_train = np.log(y_train[positive_mask_train].clip(lower=1e-9))
            y_log_train = np.nan_to_num(y_log_train, posinf=0, neginf=0)
            model.fit(X_log_train, y_log_train)
            reg_step = model.named_steps['linearregression']
            coefficients = reg_step.coef_
            intercept = reg_step.intercept_

        elif reg_type == "Lasso":
            steps = [LassoCV(cv=cv, random_state=42)]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            lasso_step = model.named_steps['lassocv']
            coefficients = lasso_step.coef_
            intercept = lasso_step.intercept_

        elif reg_type == "SVR (Метод опорных векторов)":
            param_grid = {
                'svr__C': [0.1, 1, 10],
                'svr__epsilon': [0.01, 0.1, 0.5],
                'svr__kernel': ['rbf', 'linear']
            }
            steps = [SVR()]
            if scaler is not None:
                steps.insert(0, scaler)
            base_model = make_pipeline(*steps)
            model = GridSearchCV(base_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "Decision Tree (Решающее дерево)":
            param_grid = {
                'max_depth': [None, 3, 5, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = GridSearchCV(DecisionTreeRegressor(random_state=42), param_grid, cv=cv,
                                 scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "Random Forest (Случайный лес)":
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
            model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=cv,
                                 scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "Gradient Boosting (Градиентный бустинг)":
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5]
            }
            model = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=cv,
                                 scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "HistGradientBoosting (Быстрый градиентный бустинг)":
            param_grid = {
                'learning_rate': [0.01, 0.1],
                'max_iter': [100, 200],
                'max_depth': [None, 10]
            }
            model = GridSearchCV(HistGradientBoostingRegressor(random_state=42), param_grid, cv=cv,
                                 scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "XGBoost (XGBoost)":
            if not XGBOOST_AVAILABLE:
                return {"error": "XGBoost не установлен. Установите: pip install xgboost"}
            param_grid = {
                'xgbregressor__n_estimators': [50, 100],
                'xgbregressor__max_depth': [3, 6],
                'xgbregressor__learning_rate': [0.01, 0.1],
                'xgbregressor__subsample': [0.8, 1.0]
            }
            steps = [XGBRegressor(objective='reg:squarederror', random_state=42)]
            if scaler is not None:
                steps.insert(0, scaler)
            base_model = make_pipeline(*steps)
            model = GridSearchCV(base_model, param_grid, cv=cv, scoring='neg_mean_squared_error')
            model.fit(X_train, y_train)
            model = model.best_estimator_
            coefficients = None
            intercept = None

        elif reg_type == "Gaussian Processes (Гауссовские процессы)":
            kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))
            steps = [GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=10)]
            if scaler is not None:
                steps.insert(0, scaler)
            model = make_pipeline(*steps)
            model.fit(X_train, y_train)
            coefficients = None
            intercept = None

        elif reg_type == "Neural Network (Нейронная сеть)":
            scaler = StandardScaler() if scaler is None else scaler
            X_train_scaled = scaler.fit_transform(X_train)
            model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_scaled.shape[1]),
                Dense(32, activation='relu'),
                Dense(1)
            ])
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(
                X_train_scaled, y_train,
                epochs=100, batch_size=16, validation_split=0.2,
                callbacks=[early_stopping], verbose=0
            )
            model.scaler = scaler
            coefficients = None
            intercept = None

        else:
            return {"error": f"Тип регрессии '{reg_type}' не поддерживается"}

        if reg_type in ["Логарифмическая"]:
            X_log_test = np.log(X_test[positive_mask_test].clip(lower=1e-9))
            X_log_test = np.nan_to_num(X_log_test, posinf=0, neginf=0)
            y_pred = model.predict(X_log_test)
            y_test_vals = y_test[positive_mask_test].values
        elif reg_type == "Экспоненциальная":
            y_pred = np.exp(model.predict(X_test[positive_mask_test]))
            y_test_vals = y_test[positive_mask_test].values
        elif reg_type == "Степенная":
            X_log_test = np.log(X_test[positive_mask_test].clip(lower=1e-9))
            X_log_test = np.nan_to_num(X_log_test, posinf=0, neginf=0)
            y_pred = np.exp(model.predict(X_log_test))
            y_test_vals = y_test[positive_mask_test].values
        elif reg_type == "Neural Network (Нейронная сеть)":
            X_test_scaled = model.scaler.transform(X_test)
            y_pred = model.predict(X_test_scaled).flatten()
            y_test_vals = y_test.values
        else:
            y_pred = model.predict(X_test)
            y_test_vals = y_test.values

        y_pred_vals = y_pred.flatten() if hasattr(y_pred, 'flatten') else np.array(y_pred)

        if len(y_test_vals) != len(y_pred_vals):
            raise ValueError(f"Размерности не совпадают: y_test={len(y_test_vals)}, y_pred={len(y_pred_vals)}")

        r2 = r2_score(y_test_vals, y_pred_vals)
        color = "🟢" if r2 > 0.7 else "🟡" if r2 > 0.5 else "🔴"

        return {
            "model": model,
            "metrics": {
                "r2": r2 if r2 is not None else 0,
                "mse": mean_squared_error(y_test_vals, y_pred_vals) if y_test_vals.size > 0 else 0,
                "rmse": np.sqrt(mean_squared_error(y_test_vals, y_pred_vals)) if y_test_vals.size > 0 else 0,
                "mae": mean_absolute_error(y_test_vals, y_pred_vals) if y_test_vals.size > 0 else 0,
                "mape": mean_absolute_percentage_error(y_test_vals, y_pred_vals) if y_test_vals.size > 0 else 0
            },
            "coefficients": coefficients,
            "intercept": intercept,
            "feature_names": feature_names,
            "original_features": original_features,
            "y_test": y_test_vals,
            "y_pred": y_pred_vals,
            "regression_type": reg_type,
            "color": color,
            "scaling_method": scaling_method
        }

    except Exception as e:
        logger.error(f"Ошибка при обучении модели {reg_type}: {str(e)}")
        return {"error": str(e)}


def run_selected_regressions(df, scaling_method, selected_models):
    results = {}
    feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]
    if len(feature_cols) == 0:
        st.error("Нет признаков для анализа")
        return results, None, None

    X = df[feature_cols]
    y = df["A"]

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except Exception as e:
        st.error(f"Ошибка при разделении данных: {str(e)}")
        return results, None, None

    try:
        X_train_np = X_train.to_numpy(dtype=np.float64)
        y_train_np = y_train.to_numpy(dtype=np.float64)
        X_test_np = X_test.to_numpy(dtype=np.float64)
        y_test_np = y_test.to_numpy(dtype=np.float64)
        X_train_np = np.nan_to_num(X_train_np, nan=0.0)
        y_train_np = np.nan_to_num(y_train_np, nan=0.0)
        X_test_np = np.nan_to_num(X_test_np, nan=0.0)
        y_test_np = np.nan_to_num(y_test_np, nan=0.0)
        positive_mask_train = (X_train_np > 0).all(axis=1) & (y_train_np > 0)
        positive_mask_test = (X_test_np > 0).all(axis=1) & (y_test_np > 0)
        positive_data_available = positive_mask_train.any()
    except Exception as e:
        st.error(f"Ошибка при обработке данных: {str(e)}")
        positive_mask_train = np.zeros(len(X_train), dtype=bool)
        positive_mask_test = np.zeros(len(X_test), dtype=bool)
        positive_data_available = False

    regression_types = [
        "Линейная", "Квадратическая", "Кубическая", "Логарифмическая",
        "Экспоненциальная", "Степенная", "Lasso", "SVR (Метод опорных векторов)",
        "Decision Tree (Решающее дерево)", "Random Forest (Случайный лес)",
        "Gradient Boosting (Градиентный бустинг)", "HistGradientBoosting (Быстрый градиентный бустинг)",
        "XGBoost (XGBoost)", "Gaussian Processes (Гауссовские процессы)",
        "Neural Network (Нейронная сеть)"
    ]

    progress_bar = st.progress(0)
    status_text = st.empty()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}
        for i, reg_type in enumerate(regression_types):
            if reg_type not in selected_models:
                continue
            if reg_type in ["Логарифмическая", "Экспоненциальная", "Степенная"] and not positive_data_available:
                results[reg_type] = {"error": "Требуются положительные значения"}
                progress_bar.progress((i + 1) / len(selected_models))
                status_text.text(f"Обработано {i + 1} из {len(selected_models)} моделей")
                continue
            future = executor.submit(
                train_model,
                reg_type, X_train, y_train, X_test, y_test,
                feature_cols, positive_mask_train, positive_mask_test, scaling_method
            )
            futures[future] = reg_type
        for i, future in enumerate(as_completed(futures)):
            reg_type = futures[future]
            try:
                result = future.result()
                results[reg_type] = result
                progress_bar.progress((i + 1) / len(selected_models))
                status_text.text(f"Обработано {i + 1} из {len(selected_models)} моделей")
            except Exception as e:
                results[reg_type] = {"error": str(e)}
                logger.error(f"Ошибка при обучении модели {reg_type}: {str(e)}")

    return results, X_train, y_train


def evaluate_with_cross_validation(model, X, y, model_name):
    try:
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
        return {
            "model_name": model_name,
            "mean_r2": np.mean(scores),
            "std_r2": np.std(scores),
            "all_scores": scores
        }
    except Exception as e:
        logger.error(f"Ошибка при кросс-валидации модели {model_name}: {str(e)}")
        return None


def show_regression_results(results, X_train, y_train):
    st.subheader("Сравнение моделей по R²")
    comparison_data = []
    for reg_type, res in results.items():
        if "error" not in res:
            # Добавляем проверку на None перед добавлением в comparison_data
            r2 = res["metrics"]["r2"] if res["metrics"]["r2"] is not None else 0
            rmse = res["metrics"]["rmse"] if res["metrics"]["rmse"] is not None else 0
            mae = res["metrics"]["mae"] if res["metrics"]["mae"] is not None else 0
            mape = res["metrics"]["mape"] if res["metrics"]["mape"] is not None else 0

            comparison_data.append({
                "Модель": f"{res['color']} {reg_type}",
                "R²": res["metrics"]["r2"],
                "Среднеквадратичная ошибка (RMSE):": res["metrics"]["rmse"],
                "Средняя абсолютная ошибка (MAE):": res["metrics"]["mae"],
                "Средняя абсолютная процентная ошибка (MAPE):": res["metrics"]["mape"]
            })
    if not comparison_data:
        st.error("Ни одна модель не была успешно обучена")
        return
    comparison_df = pd.DataFrame(comparison_data).sort_values("R²", ascending=False)

    def style_row(row):
        styles = [''] * len(row)
        r2 = row['R²']
        if r2 > 0.7:
            styles[1] = 'background-color: #4CAF50; color: white'
        elif r2 > 0.5:
            styles[1] = 'background-color: #FFC107; color: black'
        else:
            styles[1] = 'background-color: #F44336; color: white'
        return styles

    styled_df = comparison_df.style.apply(style_row, axis=1).format({
        'R²': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}', 'MAPE': '{:.2%}'
    })
    st.dataframe(styled_df)
    fig = px.bar(comparison_df, x='Модель', y='R²', color='R²',
                 color_continuous_scale=['#F44336', '#FFC107', '#4CAF50'], range_color=[0, 1],
                 title='Сравнение моделей по R²')
    st.plotly_chart(fig, use_container_width=True, key=f"models_comparison_{uuid.uuid4()}")
    st.markdown('<div id="top-anchor"></div>', unsafe_allow_html=True)
    st.markdown("""
    <style>
    .top-button {
        position: fixed; bottom: 30px; right: 30px; z-index: 9999; background-color: #4CAF50; color: white;
        padding: 12px 18px; border-radius: 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.2); cursor: pointer;
        font-size: 14px; font-weight: bold; border: none; transition: background-color 0.3s;
    }
    .top-button:hover { background-color: #45a049; }
    </style>
    <a href="#top-anchor" style="text-decoration: none;"><button class="top-button">Наверх</button></a>
    """, unsafe_allow_html=True)
    st.subheader("Детальные результаты по моделям")
    for reg_type, res in results.items():
        model_title = f"{res['color']} {reg_type}" if "error" not in res else reg_type
        with st.expander(f"Модель: {model_title}", expanded=False):
            if "error" in res:
                st.error(f"Ошибка: {res['error']}")
                continue
            st.write(f"**R²:** {res['metrics']['r2']:.4f}")
            st.write(f"**Среднеквадратичная ошибка (RMSE):** {res['metrics']['rmse']:.4f}")
            st.write(f"**Средняя абсолютная ошибка (MAE):** {res['metrics']['mae']:.4f}")
            st.write(f"**Средняя абсолютная процентная ошибка (MAPE):** {res['metrics']['mape']:.2%}")
            linear_models = ["Линейная", "Квадратическая", "Кубическая", "Lasso", "Логарифмическая", "Экспоненциальная",
                             "Степенная"]
            if reg_type in linear_models and res["coefficients"] is not None:
                try:
                    X_for_p = X_train
                    y_for_p = y_train
                    if reg_type in ["Квадратическая", "Кубическая"]:
                        poly = res["model"].named_steps['polynomialfeatures']
                        X_poly = poly.transform(X_train)
                        X_for_p = pd.DataFrame(X_poly, index=X_train.index,
                                               columns=poly.get_feature_names_out(X_train.columns))
                        mask = X_for_p.apply(lambda row: np.isfinite(row).all(), axis=1) & np.isfinite(y_for_p)
                        X_for_p = X_for_p[mask]
                        y_for_p = y_for_p[mask]
                    elif reg_type == "Логарифмическая":
                        X_for_p = np.log(X_train.clip(lower=1e-9))
                        X_for_p = np.nan_to_num(X_for_p, posinf=0, neginf=0)
                    elif reg_type == "Экспоненциальная":
                        y_for_p = np.log(y_train.clip(lower=1e-9))
                        y_for_p = np.nan_to_num(y_for_p, posinf=0, neginf=0)
                    elif reg_type == "Степенная":
                        X_for_p = np.log(X_train.clip(lower=1e-9))
                        X_for_p = np.nan_to_num(X_for_p, posinf=0, neginf=0)
                        y_for_p = np.log(y_train.clip(lower=1e-9))
                        y_for_p = np.nan_to_num(y_for_p, posinf=0, neginf=0)
                    mask = np.isfinite(X_for_p).all(axis=1) & np.isfinite(y_for_p)
                    X_for_p = X_for_p[mask]
                    y_for_p = y_for_p[mask]
                    p_values = show_model_significance(X_for_p, y_for_p, res["model"])
                    show_formula(
                        res["coefficients"],
                        res["intercept"],
                        res["feature_names"],
                        reg_type,
                        p_values=p_values,
                        model_pipeline=res["model"],
                        result=res
                    )
                    show_feature_importance(res["coefficients"], res["feature_names"], p_values)
                except Exception as e:
                    st.warning(f"Не удалось показать статистическую значимость: {str(e)}")
            else:
                show_feature_importance(res["coefficients"], res["feature_names"], None, res["model"], X_train, y_train)
            plot_actual_vs_predicted(res["y_test"], res["y_pred"], reg_type)
            plot_residuals(res["y_test"], res["y_pred"], reg_type)
            if len(res["original_features"]) >= 2:
                plot_response_surface(res["model"], X_train, y_train, res["original_features"], res["regression_type"],
                                      model_key=reg_type.replace(" ", "_"))

            st.markdown("---")
            analytical_models = [
                "Линейная", "Квадратическая", "Кубическая",
                "Логарифмическая", "Экспоненциальная", "Степенная", "Lasso"
            ]

            if reg_type not in analytical_models:
                save_trained_model(res["model"], reg_type)
            else:
                st.info("Аналитические модели представлены формулой. Сохранение модели не требуется.")


def data_preparation(df):
    st.subheader("Анализ выбросов")
    if df is None:
        return df

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    total_outliers = 0
    outlier_info = []

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        n_outliers = len(outliers)
        total_outliers += n_outliers
        if n_outliers > 0:
            outlier_info.append(f"- `{col}`: {n_outliers} выбросов (границы: {lower_bound:.3f} – {upper_bound:.3f})")

    if total_outliers > 0:
        st.warning(f"Обнаружено **{total_outliers} выбросов** в следующих столбцах:")
        for line in outlier_info:
            st.write(line)
    else:
        st.success("Выбросы не обнаружены (по методу IQR).")

    return df


def main():
    st.title("Корреляционно-регрессионный анализ, оптимизация, обратная оптимизация")

    # Инициализация session_state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'vif_remove_list' not in st.session_state:
        st.session_state.vif_remove_list = []
    if 'scaler_recommendation' not in st.session_state:
        st.session_state.scaler_recommendation = {"recommended": "StandardScaler (стандартизация)"}
    if 'simulator_inputs' not in st.session_state:
        st.session_state.simulator_inputs = {}
    if 'optimization_history' not in st.session_state:
        st.session_state.optimization_history = []
    if 'optimization_result' not in st.session_state:
        st.session_state.optimization_result = None
    if 'monte_carlo_samples' not in st.session_state:
        st.session_state.monte_carlo_samples = None
    if 'monte_carlo_predictions' not in st.session_state:
        st.session_state.monte_carlo_predictions = None
    if 'sensitivity_analysis' not in st.session_state:
        st.session_state.sensitivity_analysis = None
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
    if 'selected_models' not in st.session_state:
        st.session_state.selected_models = [
            "Линейная", "Квадратическая", "Кубическая", "Lasso",
            "Random Forest (Случайный лес)", "Gradient Boosting (Градиентный бустинг)",
            "Neural Network (Нейронная сеть)"
        ]

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Загрузка данных",
        "2. Анализ данных",
        "3. Корреляционный анализ",
        "4. Регрессионный анализ",
        "5. Оптимизация"
    ])

    with tab1:
        st.header("Загрузка и проверка данных")
        df = load_data()
        if df is not None:
            st.session_state.df = df.copy()
            st.session_state.processed_df = df.copy()
            st.success(f"Данные успешно загружены! Количество строк: {len(df)}, столбцов: {len(df.columns)}")
        else:
            st.session_state.df = None
            st.session_state.processed_df = None

    with tab2:
        st.header("Анализ исходных данных")
        if st.session_state.df is not None:
            show_descriptive_analysis(st.session_state.df)
            st.session_state.processed_df = data_preparation(st.session_state.df)
            st.info("Данные будут обработаны на вкладке 'Корреляционный анализ' после выбора метода.")
        else:
            st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'")

    with tab3:
        st.header("Корреляционный анализ")
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'.")
            return

        df_original = st.session_state.df.copy()
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        target_col = 'A'
        if target_col not in df_original.columns:
            st.error("Отсутствует целевая переменная 'A'")
            return

        st.subheader("Анализ выбросов")
        total_outliers = 0
        outlier_info = []
        for col in numeric_cols:
            Q1 = df_original[col].quantile(0.25)
            Q3 = df_original[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_original[(df_original[col] < lower_bound) | (df_original[col] > upper_bound)]
            n_outliers = len(outliers)
            total_outliers += n_outliers
            if n_outliers > 0:
                outlier_info.append(
                    f"- `{col}`: {n_outliers} выбросов (границы: {lower_bound:.3f} – {upper_bound:.3f})")
        if total_outliers > 0:
            st.warning(f"Обнаружено **{total_outliers} выбросов** в следующих столбцах:")
            for line in outlier_info:
                st.write(line)
        else:
            st.success("Выбросы не обнаружены (по методу IQR).")

        st.subheader("Выбор метода обработки выбросов")
        outlier_method = st.radio(
            "Выберите способ обработки выбросов:",
            ["Без обработки", "Замена границами (IQR)", "Удаление строк с выбросами"],
            help="IQR: межквартильный размах (1.5 * IQR)",
            key="outlier_method_radio"
        )

        df_processed = df_original.copy()
        total_modified = 0
        if outlier_method != "Без обработки":
            for col in numeric_cols:
                Q1 = df_original[col].quantile(0.25)
                Q3 = df_original[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers_mask = (df_original[col] < lower_bound) | (df_original[col] > upper_bound)
                n_outliers = outliers_mask.sum()
                if n_outliers == 0:
                    continue
                if outlier_method == "Замена границами (IQR)":
                    df_processed[col] = df_processed[col].clip(lower=lower_bound, upper=upper_bound)
                    total_modified += n_outliers
                elif outlier_method == "Удаление строк с выбросами":
                    indices_to_drop = outliers_mask[outliers_mask].index.intersection(df_processed.index)
                    df_processed = df_processed.drop(index=indices_to_drop)
                    total_modified += len(indices_to_drop)
            if total_modified > 0:
                st.info(f"Метод: **{outlier_method}**. Обработано выбросов: {total_modified}")
        else:
            st.session_state.processed_df = df_processed.copy()

        st.subheader("🔍 Сравнение: Исходные vs Обработанные данные")
        with st.expander("📊 Объяснение корреляционного анализа", expanded=False):
            st.markdown("""
            1. Корреляция Пирсона:
            - Мера линейной зависимости между переменными
            - Диапазон значений: от -1 (полная обратная связь) до +1 (полная прямая связь)
            - Чувствителен к выбросам
            - Интерпретация: 
              * 0.9-1.0 - очень сильная
              * 0.7-0.9 - сильная
              * 0.5-0.7 - умеренная
              * 0.3-0.5 - слабая
              * 0-0.3 - очень слабая/отсутствует
            2. Корреляция Спирмана:
            - Мера монотонной зависимости (не обязательно линейной)
            - Ранговый коэффициент корреляции
            - Устойчив к выбросам
            - Используется для нелинейных зависимостей
            3. Корреляционное отношение η²:
            - Мера нелинейной зависимости между переменными
            - Диапазон: 0-1 (1 - полная зависимость)
            - Показывает долю дисперсии Y, объясненную X
            - Учитывает любые формы зависимости
            4. VIF (Фактор инфляции дисперсии):
            - Мера мультиколлинеарности
            - Показывает насколько дисперсия коэффициента увеличена из-за корреляции с другими предикторами
            - Интерпретация:
              * VIF = 1 - нет мультиколлинеарности
              * VIF > 5 - умеренная
              * VIF > 10 - серьезная проблема

            **Обозначения уровня значимости:**
            - \*** p < 0.001 — очень высокая значимость
            - \** p < 0.01 — высокая значимость
            - \* p < 0.05 — статистически значимо
            ⚠️ Пояснение:
            Звёздочки (*) у коэффициентов показывают, насколько признак статистически значимо влияет на целевую переменную.
            Чем меньше ( p )-значение, тем сильнее доказательства против гипотезы "коэффициент равен нулю".
            """)

        feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df_original.columns]
        compare_cols = sorted([target_col] + feature_cols)

        def add_stars_to_corr(corr_matrix, p_matrix):
            sig_matrix = corr_matrix.copy().astype(str)
            for i in range(corr_matrix.shape[0]):
                for j in range(corr_matrix.shape[1]):
                    if i == j:
                        sig_matrix.iloc[i, j] = ""
                        continue
                    corr = corr_matrix.iloc[i, j]
                    p = p_matrix.iloc[i, j]
                    if pd.isna(p) or pd.isna(corr):
                        sig_matrix.iloc[i, j] = "NaN"
                        continue
                    if p < 0.001:
                        sig = f"{corr:.2f}***"
                    elif p < 0.01:
                        sig = f"{corr:.2f}**"
                    elif p < 0.05:
                        sig = f"{corr:.2f}*"
                    else:
                        sig = f"{corr:.2f}"
                    sig_matrix.iloc[i, j] = sig
            return sig_matrix

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟦 Корреляция Пирсона (исходные)")
            corr_orig = df_original[compare_cols].corr(method='pearson').round(2)
            p_orig = calculate_pvalues(df_original[compare_cols], method='pearson')
            corr_orig_stars = add_stars_to_corr(corr_orig, p_orig)
            np.fill_diagonal(corr_orig.values, np.nan)
            fig1 = px.imshow(
                corr_orig,
                text_auto=False,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title="Пирсон (до)",
                width=700, height=600
            )
            fig1.update_traces(text=corr_orig_stars.values, texttemplate="%{text}", textfont={"size": 14})
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.markdown("### 🟩 Корреляция Пирсона (обработанные)")
            corr_proc = df_processed[compare_cols].corr(method='pearson').round(2)
            p_proc = calculate_pvalues(df_processed[compare_cols], method='pearson')
            corr_proc_stars = add_stars_to_corr(corr_proc, p_proc)
            np.fill_diagonal(corr_proc.values, np.nan)
            fig2 = px.imshow(
                corr_proc,
                text_auto=False,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title="Пирсон (после)",
                width=700, height=600
            )
            fig2.update_traces(text=corr_proc_stars.values, texttemplate="%{text}", textfont={"size": 14})
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🟦 Корреляция Спирмана (исходные)")
            corr_orig_s = df_original[compare_cols].corr(method='spearman').round(2)
            p_orig_s = calculate_pvalues(df_original[compare_cols], method='spearman')
            corr_orig_s_stars = add_stars_to_corr(corr_orig_s, p_orig_s)
            np.fill_diagonal(corr_orig_s.values, np.nan)
            fig3 = px.imshow(
                corr_orig_s,
                text_auto=False,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title="Спирман (до)",
                width=700, height=600
            )
            fig3.update_traces(text=corr_orig_s_stars.values, texttemplate="%{text}", textfont={"size": 14})
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            st.markdown("### 🟩 Корреляция Спирмана (обработанные)")
            corr_proc_s = df_processed[compare_cols].corr(method='spearman').round(2)
            p_proc_s = calculate_pvalues(df_processed[compare_cols], method='spearman')
            corr_proc_s_stars = add_stars_to_corr(corr_proc_s, p_proc_s)
            np.fill_diagonal(corr_proc_s.values, np.nan)
            fig4 = px.imshow(
                corr_proc_s,
                text_auto=False,
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1,
                title="Спирман (после)",
                width=700, height=600
            )
            fig4.update_traces(text=corr_proc_s_stars.values, texttemplate="%{text}", textfont={"size": 14})
            st.plotly_chart(fig4, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 🔗 η² (исходные)")
            eta_orig = calculate_correlation_ratio(df_original, target_col)
            eta_df_orig = pd.DataFrame.from_dict(eta_orig, orient='index', columns=['η²']).round(3)
            eta_df_orig = eta_df_orig.loc[feature_cols] if feature_cols else eta_df_orig
            fig7 = px.bar(eta_df_orig, y='η²', color='η²', range_color=[0, 1], text='η²', title="η² (до)")
            fig7.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig7, use_container_width=True)

        with col2:
            st.markdown("### 🟢 η² (обработанные)")
            eta_proc = calculate_correlation_ratio(df_processed, target_col)
            eta_df_proc = pd.DataFrame.from_dict(eta_proc, orient='index', columns=['η²']).round(3)
            eta_df_proc = eta_df_proc.loc[feature_cols] if feature_cols else eta_df_proc
            fig8 = px.bar(eta_df_proc, y='η²', color='η²', range_color=[0, 1], text='η²', title="η² (после)")
            fig8.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig8, use_container_width=True)

        col1, col2 = st.columns(2)
        st.subheader("📊 Анализ мультиколлинеарности (VIF)")

        all_features = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df_original.columns]

        if all_features:
            # Инициализация session_state для исключенных признаков
            if 'vif_excluded_features' not in st.session_state:
                st.session_state.vif_excluded_features = []

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📊 Исходный VIF (все признаки)")
                vif_orig_all = calculate_vif(df_original[all_features])
                fig5 = px.bar(vif_orig_all, x='feature', y='VIF', color='VIF',
                              color_continuous_scale=['green', 'orange', 'red'], range_color=[0, 20],
                              text='VIF', title="VIF - все признаки (до обработки)")
                fig5.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 - высокая")
                fig5.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="VIF=5 - умеренная")
                fig5.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig5, use_container_width=True)

                # Статистика по исходному VIF
                high_vif_count_orig = (vif_orig_all['VIF'] >= 10).sum()
                moderate_vif_count_orig = ((vif_orig_all['VIF'] >= 5) & (vif_orig_all['VIF'] < 10)).sum()

                st.info(f"""
                    **Статистика исходного VIF:**
                    - Всего признаков: **{len(all_features)}**
                    - VIF ≥ 10 (высокая): **{high_vif_count_orig}**
                    - 5 ≤ VIF < 10 (умеренная): **{moderate_vif_count_orig}**
                    - VIF < 5 (низкая): **{len(all_features) - high_vif_count_orig - moderate_vif_count_orig}**
                    """)

            with col2:
                st.markdown("### 📈 VIF с исключением признаков")

                # Мультиселект для исключения признаков
                excluded_features = st.multiselect(
                    "Исключить признаки из анализа VIF:",
                    options=all_features,
                    default=st.session_state.vif_excluded_features,
                    key="vif_exclude_selector"
                )

                st.session_state.vif_excluded_features = excluded_features

                # Признаки для анализа после исключения
                remaining_features = [f for f in all_features if f not in excluded_features]

                if not remaining_features:
                    st.warning("Выберите меньше признаков для исключения")
                    remaining_features = all_features[:min(3, len(all_features))]  # Минимум 3 признака
                    st.info(f"Будут использованы: {', '.join(remaining_features)}")

                st.write(f"**Анализируемые признаки ({len(remaining_features)}):** {', '.join(remaining_features)}")

                if excluded_features:
                    st.write(f"**Исключенные признаки ({len(excluded_features)}):** {', '.join(excluded_features)}")

                # Расчет VIF для оставшихся признаков
                vif_remaining = calculate_vif(df_original[remaining_features])

                fig6 = px.bar(vif_remaining, x='feature', y='VIF', color='VIF',
                              color_continuous_scale=['green', 'orange', 'red'], range_color=[0, 20],
                              text='VIF', title=f"VIF после исключения {len(excluded_features)} признаков")
                fig6.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="VIF=10 - высокая")
                fig6.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="VIF=5 - умеренная")
                fig6.update_traces(texttemplate='%{text:.1f}', textposition='outside')
                st.plotly_chart(fig6, use_container_width=True)

                # Статистика по VIF после исключения
                high_vif_count_remaining = (vif_remaining['VIF'] >= 10).sum()
                moderate_vif_count_remaining = ((vif_remaining['VIF'] >= 5) & (vif_remaining['VIF'] < 10)).sum()

                st.info(f"""
                    **Статистика после исключения:**
                    - Осталось признаков: **{len(remaining_features)}**
                    - VIF ≥ 10 (высокая): **{high_vif_count_remaining}**
                    - 5 ≤ VIF < 10 (умеренная): **{moderate_vif_count_remaining}**
                    - VIF < 5 (низкая): **{len(remaining_features) - high_vif_count_remaining - moderate_vif_count_remaining}**
                    """)

                # Анализ улучшения
                if excluded_features:
                    improvement = high_vif_count_orig - high_vif_count_remaining
                    if improvement > 0:
                        st.success(f"✅ Улучшение: уменьшено признаков с VIF≥10 на {improvement}")
                    elif high_vif_count_remaining == 0:
                        st.success("✅ Отличный результат! Нет признаков с высокой мультиколлинеарностью")
                    else:
                        st.warning("⚠️ Мультиколлинеарность все еще высокая. Попробуйте исключить другие признаки")

            # Сравнительный анализ
            st.subheader("📊 Сравнительный анализ VIF")

            col1, col2 = st.columns(2)

            with col1:
                # Сводная таблица сравнения
                comparison_data = {
                    'Метрика': ['Всего признаков', 'VIF ≥ 10', '5 ≤ VIF < 10', 'VIF < 5'],
                    'Исходно': [
                        len(all_features),
                        high_vif_count_orig,
                        moderate_vif_count_orig,
                        len(all_features) - high_vif_count_orig - moderate_vif_count_orig
                    ],
                    'После исключения': [
                        len(remaining_features),
                        high_vif_count_remaining,
                        moderate_vif_count_remaining,
                        len(remaining_features) - high_vif_count_remaining - moderate_vif_count_remaining
                    ]
                }

                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(
                    comparison_df.style.format({
                        'Исходно': '{:.0f}',
                        'После исключения': '{:.0f}'
                    }).apply(lambda x: [''] * len(x), axis=1)
                )

            with col2:
                # Визуализация улучшения
                if excluded_features:
                    fig_improve = go.Figure()

                    fig_improve.add_trace(go.Bar(
                        name='Исходно',
                        x=['VIF ≥ 10', '5 ≤ VIF < 10', 'VIF < 5'],
                        y=[high_vif_count_orig, moderate_vif_count_orig,
                           len(all_features) - high_vif_count_orig - moderate_vif_count_orig],
                        marker_color='blue'
                    ))

                    fig_improve.add_trace(go.Bar(
                        name='После исключения',
                        x=['VIF ≥ 10', '5 ≤ VIF < 10', 'VIF < 5'],
                        y=[high_vif_count_remaining, moderate_vif_count_remaining,
                           len(remaining_features) - high_vif_count_remaining - moderate_vif_count_remaining],
                        marker_color='green'
                    ))

                    fig_improve.update_layout(
                        title="Сравнение распределения VIF",
                        barmode='group',
                        showlegend=True
                    )

                    st.plotly_chart(fig_improve, use_container_width=True)

            # Рекомендации по оптимизации
            st.subheader("💡 Рекомендации по оптимизации мультиколлинеарности")

            # Автоматические рекомендации какие признаки исключить
            if high_vif_count_remaining > 0:
                high_vif_features = vif_remaining[vif_remaining['VIF'] >= 10]['feature'].tolist()
                st.warning(f"**Рекомендуется исключить:** {', '.join(high_vif_features)}")

                # Кнопка для автоматического применения рекомендаций
                if st.button("🗑️ Применить рекомендации и исключить признаки с VIF≥10",
                             key="apply_vif_recommendations"):
                    st.session_state.vif_excluded_features.extend(high_vif_features)
                    st.session_state.vif_excluded_features = list(set(st.session_state.vif_excluded_features))
                    st.success(f"Добавлены к исключению: {', '.join(high_vif_features)}")
                    st.rerun()

            # Кнопка сброса
            if st.button("🔄 Сбросить исключения", key="reset_vif_exclusions"):
                st.session_state.vif_excluded_features = []
                st.success("Исключения сброшены")
                st.rerun()

            # Перенос исключенных признаков в список для удаления в регрессии
            if excluded_features and st.button("📋 Перенести исключенные признаки в список удаления",
                                               key="transfer_to_remove"):
                st.session_state.vif_remove_list.extend(excluded_features)
                st.session_state.vif_remove_list = list(set(st.session_state.vif_remove_list))
                st.success(f"Признаки добавлены в список для удаления: {', '.join(excluded_features)}")

        else:
            st.info("Нет признаков для VIF анализа.")

    with tab4:
        st.header("Регрессионный анализ")
        if st.session_state.processed_df is not None:
            if "A" not in st.session_state.processed_df.columns:
                st.error("Отсутствует целевая переменная 'A'")
            else:
                df = st.session_state.processed_df.copy()
                feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]

                st.subheader("Удаление признаков с высокой мультиколлинеарностью (VIF ≥ 10)")
                if len(feature_cols) > 0:
                    X_vif = df[feature_cols]
                    vif_data = calculate_vif(X_vif)
                    high_vif = vif_data[vif_data['VIF'] >= 10]

                    if not high_vif.empty:
                        st.warning("Обнаружены признаки с высокой мультиколлинеарностью (VIF ≥ 10):")
                        for _, row in high_vif.iterrows():
                            checked = st.checkbox(
                                f"Удалить '{row['feature']}' (VIF = {row['VIF']:.2f})",
                                value=(row['feature'] in st.session_state.vif_remove_list),
                                key=f"vif_remove_{row['feature']}"
                            )
                            if checked and row['feature'] not in st.session_state.vif_remove_list:
                                st.session_state.vif_remove_list.append(row['feature'])
                            elif not checked and row['feature'] in st.session_state.vif_remove_list:
                                st.session_state.vif_remove_list.remove(row['feature'])
                        st.info(f"Помечены к удалению: {', '.join(st.session_state.vif_remove_list)}")
                    else:
                        st.success("Значительной мультиколлинеарности не обнаружено (VIF < 10 для всех признаков)")
                        st.session_state.vif_remove_list = []
                else:
                    st.warning("Нет признаков для анализа VIF.")
                    st.session_state.vif_remove_list = []

                st.write("### Выбор метода масштабирования")
                st.markdown("""
                - **StandardScaler (стандартизация)**: Среднее = 0, std = 1. Подходит для большинства моделей.
                - **MinMaxScaler (нормализация)**: Диапазон [0, 1]. Рекомендуется для нейросетей.
                - **RobustScaler (устойчивый)**: Устойчив к выбросам.
                - **Нет**: Без масштабирования.
                """)
                scaling_method = st.selectbox(
                    "Выберите метод масштабирования:",
                    ["Нет", "StandardScaler (стандартизация)", "MinMaxScaler (нормализация)",
                     "RobustScaler (устойчивый)"],
                    index=["Нет", "StandardScaler (стандартизация)", "MinMaxScaler (нормализация)",
                           "RobustScaler (устойчивый)"]
                    .index(st.session_state.get('scaler_recommendation', {}).get('recommended', 'StandardScaler')),
                    key="scaling_method_regression"
                )

                # Новый аккордеон для выбора моделей с группировкой
                with st.expander("Выбор моделей для обучения", expanded=True):
                    st.write("Выберите модели из групп (галочки применяются после нажатия кнопки):")

                    # Группы моделей
                    linear_models = {
                        "Линейная": "Простая линейная регрессия",
                        "Квадратическая": "Полиномиальная (степень 2)",
                        "Кубическая": "Полиномиальная (степень 3)",
                        "Lasso": "L1-регуляризация (автоматический отбор признаков)"
                    }

                    nonlinear_models = {
                        "Логарифмическая": "Линейная по логарифмам признаков",
                        "Экспоненциальная": "Экспоненциальная зависимость",
                        "Степенная": "Степенная зависимость (y = a*x^b)"
                    }

                    tree_models = {
                        "Decision Tree (Решающее дерево)": "Дерево решений",
                        "Random Forest (Случайный лес)": "Ансамбль деревьев",
                        "Gradient Boosting (Градиентный бустинг)": "Последовательное улучшение предсказаний",
                        "HistGradientBoosting (Быстрый градиентный бустинг)": "Оптимизированная версия"
                    }

                    other_models = {
                        "SVR (Метод опорных векторов)": "SVM для регрессии",
                        "XGBoost (XGBoost)": "Эффективный градиентный бустинг",
                        "Gaussian Processes (Гауссовские процессы)": "Байесовский подход",
                        "Neural Network (Нейронная сеть)": "Многослойный перцептрон"
                    }

                    # Создаем чекбоксы для каждой группы
                    selected = {}

                    st.markdown("#### Линейные модели")
                    for model, desc in linear_models.items():
                        selected[model] = st.checkbox(f"{model}: {desc}",
                                                      value=model in st.session_state.selected_models,
                                                      key=f"linear_{model}")

                    st.markdown("#### Нелинейные преобразования")
                    for model, desc in nonlinear_models.items():
                        selected[model] = st.checkbox(f"{model}: {desc}",
                                                      value=model in st.session_state.selected_models,
                                                      key=f"nonlin_{model}")

                    st.markdown("#### Древовидные модели")
                    for model, desc in tree_models.items():
                        selected[model] = st.checkbox(f"{model}: {desc}",
                                                      value=model in st.session_state.selected_models,
                                                      key=f"tree_{model}")

                    st.markdown("#### Другие алгоритмы")
                    for model, desc in other_models.items():
                        selected[model] = st.checkbox(f"{model}: {desc}",
                                                      value=model in st.session_state.selected_models,
                                                      key=f"other_{model}")

                    if st.button("Применить выбор моделей", key="apply_model_selection"):
                        st.session_state.selected_models = [model for model, is_selected in selected.items() if
                                                            is_selected]
                        st.success(f"Выбрано моделей: {len(st.session_state.selected_models)}")

                # Кнопка запуска расчётов (остаётся без изменений)
                if st.button("Запустить выбранные регрессии", key="run_regressions"):
                    with st.spinner("Подготовка данных и обучение моделей..."):
                        if st.session_state.vif_remove_list:
                            available_to_remove = [f for f in st.session_state.vif_remove_list if f in df.columns]
                            if available_to_remove:
                                df = df.drop(columns=available_to_remove)
                                st.success(f"Удалены признаки: {', '.join(available_to_remove)}")
                            else:
                                st.info("Признаки с высоким VIF не выбраны для удаления.")
                        else:
                            st.info("Признаки с высоким VIF не помечены для удаления.")

                        st.session_state.processed_df = df.copy()

                        results, X_train, y_train = run_selected_regressions(df, scaling_method,
                                                                             st.session_state.selected_models)

                        st.session_state.results = results
                        st.session_state.X_train = X_train
                        st.session_state.y_train = y_train
                        st.rerun()

                if st.session_state.results is not None:
                    show_regression_results(st.session_state.results, st.session_state.X_train,
                                            st.session_state.y_train)
                else:
                    st.info("Нажмите 'Запустить выбранные регрессии', чтобы обучить модели.")

        else:
            st.warning("Пожалуйста, загрузите и обработайте данные на предыдущих вкладках.")

    with tab5:
        st.header("Оптимизация и симуляция")
        # Аккордеоны с описанием методов
        with st.expander("📌 Общий подход к анализу", expanded=True):
            st.markdown("""
                **Комбинированная стратегия:**  
                Эта вкладка объединяет три метода для исследования модели:
                1. **Симулятор** — интерактивное тестирование «что если» для ручной проверки гипотез.
                2. **Оптимизация** — автоматический поиск параметров для целевого значения A (минимум/максимум/конкретное число).
                3. **Monte Carlo** — массовый случайный перебор параметров для оценки распределения результатов.

                ▶ **Порядок работы:**  
                - Сначала используйте симулятор, чтобы понять, как параметры влияют на A.  
                - Затем примените оптимизацию для точного поиска нужного значения.  
                - Наконец, Monte Carlo покажет устойчивость решений и возможные риски.
                """)

        with st.expander("🎮 Симулятор: интерактивное тестирование", expanded=False):
            st.markdown("""
                **Что это?**  
                Инструмент для ручного изменения параметров и мгновенного просмотра предсказания модели.

                **Зачем использовать?**  
                - Проверить, как изменение одного параметра влияет на результат.  
                - Увидеть границы разумных значений («что если поставить X=1000?»).  
                - Быстро протестировать интуитивные гипотезы.

                **Пример:**  
                Если двигать ползунок параметра B, можно увидеть, что A растёт нелинейно после B=50.
                """)

        with st.expander("🔍 Оптимизация: поиск экстремумов", expanded=False):
            st.markdown("""
                **Что это?**  
                Алгоритмы для автоматического поиска параметров, которые дают **минимум, максимум или точное значение A**.

                **Зачем использовать?**  
                - Найти условия для максимальной прибыли (A → max).  
                - Определить параметры для целевого показателя (A = 100 ± 5).  
                - Обнаружить «узкие места» модели (например, минимально возможный A).

                **Как работает?**  
                Методом численной оптимизации (SciPy) ищет комбинацию параметров, удовлетворяющую условию.  
                ⚠️ **Ограничения:** Может находить локальные (не глобальные) экстремумы!
                """)

        with st.expander("🔄 Monte Carlo: оценка распределения", expanded=False):
            st.markdown("""
                **Что это?**  
                Массовый случайный эксперимент: модель запускается тысячи раз со случайными параметрами.

                **Зачем использовать?**  
                - Увидеть диапазон возможных значений A.  
                - Оценить вероятность достижения цели (например, P(A > 80)).  
                - Найти «безопасные» комбинации параметров (где A всегда в нужном диапазоне).

                **Пример вывода:**  
                «При случайных параметрах в 90% случаев A ∈ [40, 60], максимум — 72.3».

                **Отличие от оптимизации:**  
                Monte Carlo не ищет лучший вариант, а показывает статистику возможных исходов.
                """)

        if st.session_state.results is None or st.session_state.processed_df is None:
            st.warning("Пожалуйста, выполните регрессионный анализ на вкладке 'Регрессионный анализ'.")
        else:
            valid_results = {k: v for k, v in st.session_state.results.items() if "error" not in v}
            if not valid_results:
                st.warning("Нет обученных моделей для оптимизации.")
            else:
                model_options = list(valid_results.keys())
                if 'selected_opt_model' not in st.session_state:
                    st.session_state.selected_opt_model = model_options[0]
                st.session_state.selected_opt_model = st.selectbox(
                    "Выберите модель для оптимизации",
                    model_options,
                    index=model_options.index(st.session_state.selected_opt_model),
                    key="opt_model_select"
                )
                best_model_name = st.session_state.selected_opt_model
                best_result = valid_results[best_model_name]
                model = best_result["model"]
                feature_cols = best_result["original_features"]
                df = st.session_state.processed_df
                st.success(f"Используется модель: **{best_model_name}** (R² = {best_result['metrics']['r2']:.4f})")

                # Новый раздел для поиска экстремальных значений
                st.subheader("🔍 Поиск экстремальных значений")

                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("Найти минимальное A"):
                        with st.spinner("Поиск минимума..."):
                            try:
                                from scipy.optimize import minimize

                                def objective(x):
                                    X_point = pd.DataFrame([x], columns=feature_cols)
                                    return predict_with_model(model, best_model_name, X_point, best_result)

                                bounds = [(df[col].min(), df[col].max()) for col in feature_cols]
                                initial_guess = [df[col].median() for col in feature_cols]

                                result = minimize(objective, initial_guess, bounds=bounds)

                                if result.success:
                                    st.session_state.opt_result = {
                                        "type": "min",
                                        "value": result.fun,
                                        "params": dict(zip(feature_cols, result.x))
                                    }
                            except Exception as e:
                                st.error(f"Ошибка оптимизации: {str(e)}")

                with col2:
                    if st.button("Найти максимальное A"):
                        with st.spinner("Поиск максимума..."):
                            try:
                                from scipy.optimize import minimize

                                def objective(x):
                                    X_point = pd.DataFrame([x], columns=feature_cols)
                                    return -predict_with_model(model, best_model_name, X_point, best_result)

                                bounds = [(df[col].min(), df[col].max()) for col in feature_cols]
                                initial_guess = [df[col].median() for col in feature_cols]

                                result = minimize(objective, initial_guess, bounds=bounds)

                                if result.success:
                                    st.session_state.opt_result = {
                                        "type": "max",
                                        "value": -result.fun,
                                        "params": dict(zip(feature_cols, result.x))
                                    }
                            except Exception as e:
                                st.error(f"Ошибка оптимизации: {str(e)}")

                with col3:
                    target_a = st.number_input("Целевое значение A",
                                               value=float(df["A"].mean()),
                                               min_value=float(df["A"].min()),
                                               max_value=float(df["A"].max()))

                    if st.button("Найти параметры для заданного A"):
                        with st.spinner("Поиск параметров..."):
                            try:
                                from scipy.optimize import minimize

                                def objective(x):
                                    X_point = pd.DataFrame([x], columns=feature_cols)
                                    pred = predict_with_model(model, best_model_name, X_point, best_result)
                                    return (pred - target_a) ** 2

                                bounds = [(df[col].min(), df[col].max()) for col in feature_cols]
                                initial_guess = [df[col].median() for col in feature_cols]

                                result = minimize(objective, initial_guess, bounds=bounds)

                                if result.success:
                                    pred_value = predict_with_model(model, best_model_name,
                                                                    pd.DataFrame([result.x], columns=feature_cols),
                                                                    best_result)
                                    st.session_state.opt_result = {
                                        "type": "target",
                                        "target": target_a,
                                        "actual": pred_value,
                                        "params": dict(zip(feature_cols, result.x))
                                    }
                            except Exception as e:
                                st.error(f"Ошибка оптимизации: {str(e)}")

                # Отображение результатов оптимизации
                if 'opt_result' in st.session_state:
                    res = st.session_state.opt_result
                    st.subheader("Результаты оптимизации")

                    if res["type"] == "min":
                        st.success(f"Минимальное значение A: **{res['value']:.4f}**")
                    elif res["type"] == "max":
                        st.success(f"Максимальное значение A: **{res['value']:.4f}**")
                    elif res["type"] == "target":
                        st.success(f"Целевое значение A: {res['target']:.4f}")
                        st.success(f"Полученное значение A: **{res['actual']:.4f}**")

                    st.write("Параметры модели:")
                    params_df = pd.DataFrame.from_dict(res["params"], orient='index', columns=['Значение'])
                    st.dataframe(params_df.style.format("{:.4f}"))

                    # Кнопка для применения найденных параметров в симуляторе
                    if st.button("Применить параметры в симуляторе"):
                        for col, val in res["params"].items():
                            if col in st.session_state.simulator_inputs:
                                st.session_state.simulator_inputs[col] = val
                        st.rerun()

                st.subheader("🎮 Интерактивный симулятор")
                st.caption(
                    "Значения признаков ограничены диапазоном, представленным в обучающих данных. "
                    "Это предотвращает некорректные предсказания за пределами обученного диапазона."
                )

                cols = st.columns(len(feature_cols))
                inputs = {}
                for i, col in enumerate(feature_cols):
                    with cols[i % len(cols)]:
                        if col not in st.session_state.simulator_inputs:
                            st.session_state.simulator_inputs[col] = float(df[col].median())
                        value = st.number_input(
                            f"{col}",
                            value=st.session_state.simulator_inputs[col],
                            min_value=float(df[col].min()),
                            max_value=float(df[col].max()),
                            step=0.01,
                            help=f"Допустимый диапазон: от {df[col].min():.3f} до {df[col].max():.3f}",
                            key=f"sim_input_{col}"
                        )
                        st.session_state.simulator_inputs[col] = value
                        inputs[col] = value

                if st.button("🧮 Пересчитать", key="recalculate_prediction"):
                    try:
                        X_input = pd.DataFrame([inputs])
                        pred = predict_with_model(model, best_model_name, X_input, best_result)
                        st.session_state.last_prediction = pred
                        if st.session_state.last_prediction is not None:
                            st.metric("Предсказанное значение A", f"{st.session_state.last_prediction:.4f}")
                        else:
                            st.error("Не удалось получить предсказание")
                    except Exception as e:
                        st.error(f"Ошибка предсказания: {str(e)}")
                else:
                    if 'last_prediction' in st.session_state and st.session_state.last_prediction is not None:
                        st.metric("Предсказанное значение A", f"{st.session_state.last_prediction:.4f}")
                    else:
                        st.write("Нажмите **Пересчитать**, чтобы получить предсказание.")

                st.subheader("🔍 Анализ чувствительности признаков")
                st.caption(
                    "Чувствительность = |∂A/∂x_i| — насколько сильно изменяется A при изменении признака x_i.\n"
                    "Высокая чувствительность = признак критичен. Низкая = можно варьировать свободно."
                )

                if st.button("Рассчитать чувствительность", key="sensitivity_button"):
                    with st.spinner("Вычисление градиента модели..."):
                        try:
                            from scipy.optimize import approx_fprime

                            center = np.array([df[col].median() for col in feature_cols])

                            def predict_func(x):
                                X_point = pd.DataFrame([x], columns=feature_cols)
                                return predict_with_model(model, best_model_name, X_point, best_result)

                            epsilon = 1e-4
                            gradient = approx_fprime(center, predict_func, epsilon)
                            sensitivity = np.abs(gradient)
                            sens_normalized = sensitivity / (sensitivity.max() + 1e-8)

                            threshold = np.median(sensitivity)
                            classification = ["Жёсткий (критичен)" if s > threshold else "Гибкий (варьируем)" for s in
                                              sensitivity]

                            sens_df = pd.DataFrame({
                                "Признак": feature_cols,
                                "Чувствительность |∂A/∂x|": sensitivity.round(6),
                                "Нормализованная": sens_normalized.round(3),
                                "Тип": classification
                            }).sort_values("Чувствительность |∂A/∂x|", ascending=False)

                            st.session_state.sensitivity_analysis = sens_df

                            st.dataframe(sens_df.style.format({
                                "Чувствительность |∂A/∂x|": "{:.6f}",
                                "Нормализованная": "{:.3f}"
                            }).apply(lambda row: [
                                                     'background-color: #ffebee' if row[
                                                                                        'Тип'] == 'Жёсткий (критичен)' else 'background-color: #e8f5e9'
                                                 ] * len(row), axis=1))

                            fig_sens = px.bar(
                                sens_df,
                                x="Признак",
                                y="Чувствительность |∂A/∂x|",
                                color="Чувствительность |∂A/∂x|",
                                color_continuous_scale=['lightgreen', 'orange', 'red'],
                                text="Чувствительность |∂A/∂x|",
                                title="Чувствительность признаков"
                            )
                            fig_sens.update_traces(texttemplate='%{text:.6f}', textposition='outside')
                            fig_sens.add_hline(y=threshold, line_dash="dash", line_color="gray",
                                               annotation_text="Порог")
                            st.plotly_chart(fig_sens, use_container_width=True)

                            rigid = sens_df[sens_df["Тип"] == "Жёсткий (критичен)"]["Признак"].tolist()
                            flexible = sens_df[sens_df["Тип"] == "Гибкий (варьируем)"]["Признак"].tolist()
                            st.success(f"**Жёсткие:** {', '.join(rigid)}")
                            st.info(f"**Гибкие:** {', '.join(flexible)}")

                        except Exception as e:
                            st.error(f"Ошибка при расчёте чувствительности: {str(e)}")

                st.markdown("---")

                st.subheader("🔄 Monte Carlo: найти допустимые диапазоны признаков")
                st.write("Введите диапазон целевой переменной A:")
                col1, col2 = st.columns(2)
                with col1:
                    target_low = st.number_input("Минимальное A", value=float(df["A"].quantile(0.25)),
                                                 key="target_low_mc")
                with col2:
                    target_high = st.number_input("Максимальное A", value=float(df["A"].quantile(0.75)),
                                                  key="target_high_mc")

                n_samples = st.slider("Количество симуляций", min_value=1000, max_value=50000, value=10000, step=1000)

                if st.button("Запустить Monte Carlo", key="monte_carlo_run"):
                    with st.spinner("Генерация и оценка точек..."):
                        try:
                            low_bounds = [df[col].min() for col in feature_cols]
                            high_bounds = [df[col].max() for col in feature_cols]
                            samples = np.random.uniform(
                                low=low_bounds,
                                high=high_bounds,
                                size=(n_samples, len(feature_cols))
                            )
                            X_samples = pd.DataFrame(samples, columns=feature_cols)

                            preds = []
                            for i in range(n_samples):
                                pred = predict_with_model(model, best_model_name, X_samples.iloc[[i]], best_result)
                                preds.append(pred)
                            preds = np.array(preds)

                            mask = (preds >= target_low) & (preds <= target_high)
                            X_valid = X_samples[mask]
                            valid_preds = preds[mask]

                            if len(X_valid) == 0:
                                st.warning("❌ Не найдено ни одной точки, где A ∈ [{:.3f}, {:.3f}]".format(target_low,
                                                                                                          target_high))
                            else:
                                st.success(f"✅ Найдено **{len(X_valid)}** точек, где A в целевом диапазоне")

                                ranges = X_valid.agg(['min', 'max']).round(4)
                                st.write("### 🔍 Допустимые диапазоны признаков:")
                                st.dataframe(ranges)

                                st.session_state.monte_carlo_samples = X_valid.copy()
                                st.session_state.monte_carlo_predictions = valid_preds.copy()

                                fig_hist = px.histogram(
                                    valid_preds,
                                    nbins=30,
                                    title="Распределение A среди допустимых точек",
                                    labels={"value": "A", "count": "Частота"}
                                )
                                fig_hist.add_vline(x=target_low, line_dash="dash", line_color="red")
                                fig_hist.add_vline(x=target_high, line_dash="dash", line_color="red")
                                st.plotly_chart(fig_hist, use_container_width=True)

                        except Exception as e:
                            st.error(f"Ошибка Monte Carlo: {str(e)}")

                if st.session_state.monte_carlo_samples is not None and len(st.session_state.monte_carlo_samples) > 0:
                    st.markdown("---")
                    st.subheader("📊 2D Проекция допустимой области")

                    X_valid = st.session_state.monte_carlo_samples
                    valid_preds = st.session_state.monte_carlo_predictions

                    col1, col2 = st.columns(2)
                    with col1:
                        x_axis = st.selectbox("Ось X", feature_cols, index=0, key="mc_x_axis")
                    with col2:
                        y_axis = st.selectbox("Ось Y", [f for f in feature_cols if f != x_axis], index=0,
                                              key="mc_y_axis")

                    fig_2d = px.scatter(
                        X_valid,
                        x=x_axis,
                        y=y_axis,
                        color=valid_preds,
                        color_continuous_scale='Viridis',
                        labels={x_axis: x_axis, y_axis: y_axis},
                        title=f"Допустимые точки: {x_axis} vs {y_axis} (A ∈ [{target_low:.3f}, {target_high:.3f}])",
                        hover_data={X_valid.index.name or "index": X_valid.index}
                    )
                    fig_2d.update_traces(marker=dict(size=6, opacity=0.8))
                    st.plotly_chart(fig_2d, use_container_width=True)

                    if st.checkbox("Показать исходные данные  в выбранном диапазоне (полупрозрачно)", key="show_original"):
                        mask_original = (df["A"] >= target_low) & (df["A"] <= target_high)
                        fig_2d.add_scatter(
                            x=df.loc[mask_original, x_axis],
                            y=df.loc[mask_original, y_axis],
                            mode='markers',
                            marker=dict(color='gray', size=4, opacity=0.3),
                            name='Исходные данные'
                        )
                        st.plotly_chart(fig_2d, use_container_width=True)

                st.markdown("---")


if __name__ == "__main__":
    main()
