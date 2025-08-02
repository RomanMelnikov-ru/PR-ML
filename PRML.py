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
# Загружает и кэширует данные из Excel-файла, возвращает DataFrame.
@st.cache_resource(show_spinner=False)
def load_data_cached(uploaded_file):
    return pd.read_excel(uploaded_file)

# Кэширует матрицу корреляции для заданного метода (pearson/spearman).
@st.cache_data(show_spinner=False)
def calculate_correlation_matrix(df, method='pearson'):
    return df.corr(method=method)

# Кэширует расчет VIF (коэффициент инфляции дисперсии) для анализа мультиколлинеарности.
@st.cache_data(show_spinner=False)
def calculate_vif_cached(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
    vif_data["VIF"] = vif_data["VIF"].round(2)
    return vif_data

# Отображает интерфейс загрузки данных, проверяет структуру, обрабатывает пропуски и возвращает DataFrame.
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
            df = df.dropna(axis=1, how='all')
            # Проверка на наличие пропусков и обработка
            if df.isnull().values.any() or np.isinf(df.select_dtypes(include=[np.number]).values).any():
                st.warning("Обнаружены пропущенные значения или бесконечности. Будут обработаны медианой.")
                df = handle_missing_values(df)
            # Проверка наличия целевой переменной A
            if 'A' not in df.columns:
                st.error("Ошибка: В данных отсутствует столбец 'A' (целевая переменная)")
                return None
            # Признаки: B, C, D, E, F, G, H
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

# Возвращает объект scaler по строковому имени (стандартизация, нормализация и т.д.).
def get_scaler_from_name(scaling_method):
    """Возвращает объект scaler по его названию."""
    if scaling_method == "StandardScaler (стандартизация)":
        return StandardScaler()
    elif scaling_method == "MinMaxScaler (нормализация)":
        return MinMaxScaler()
    elif scaling_method == "RobustScaler (устойчивый)":
        return RobustScaler()
    else:
        return None

# Заменяет пропуски и бесконечности медианой, удаляет признаки с нулевой дисперсией.
def handle_missing_values(df):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    imputer = SimpleImputer(strategy='median')
    df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    selector = VarianceThreshold()
    selector.fit(df[numeric_cols])
    df = df.iloc[:, selector.get_support(indices=True)]
    return df

# Обрабатывает выбросы методом IQR: заменяет значения за пределами [Q1-1.5*IQR, Q3+1.5*IQR] на границы.
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

# Вычисляет и возвращает VIF для признаков (обертка над кэшированной функцией).
def calculate_vif(X):
    return calculate_vif_cached(X)

# Отображает описательную статистику, гистограммы и scatter matrix.
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

# Отображает тепловые карты корреляций (Пирсон, Спирман), η² и VIF; анализирует мультиколлинеарность.
def show_correlation_analysis(df):
    st.subheader("Анализ корреляций и мультиколлинеарности")
    if df is None or df.empty:
        st.warning("Нет данных для анализа.")
        return df

    feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]
    if 'A' not in df.columns:
        st.error("Целевая переменная 'A' отсутствует")
        return df

    corr_cols = sorted(['A'] + feature_cols)
    X_corr = df[corr_cols]

    # Pearson
    st.subheader("Тепловая карта корреляции Пирсона (на исходных данных)")
    try:
        pearson_corr = calculate_correlation_matrix(X_corr, 'pearson').round(2)
        np.fill_diagonal(pearson_corr.values, np.nan)
        pearson_p = calculate_pvalues(X_corr, method='pearson')
        pearson_sig = add_significance_stars(pearson_corr, pearson_p)
        fig_pearson = px.imshow(
            pearson_corr,
            text_auto=False,
            color_continuous_scale=px.colors.diverging.RdBu,
            zmin=-1, zmax=1,
            labels=dict(color="Коэффициент корреляции"),
            title="Корреляция Пирсона",
            width=700, height=600
        )
        fig_pearson.update_traces(text=pearson_sig, texttemplate="%{text}", textfont={"size": 14})
        st.plotly_chart(fig_pearson, use_container_width=True, key=f"pearson_corr_{uuid.uuid4()}")
    except Exception as e:
        st.error(f"Ошибка при построении карты Пирсона: {str(e)}")

    # Spearman
    st.subheader("Тепловая карта корреляции Спирмана (на исходных данных)")
    try:
        spearman_corr = calculate_correlation_matrix(X_corr, 'spearman').round(2)
        np.fill_diagonal(spearman_corr.values, np.nan)
        spearman_p = calculate_pvalues(X_corr, method='spearman')
        spearman_sig = add_significance_stars(spearman_corr, spearman_p)
        fig_spearman = px.imshow(
            spearman_corr,
            text_auto=False,
            color_continuous_scale=px.colors.diverging.RdBu,
            zmin=-1, zmax=1,
            labels=dict(color="Коэффициент корреляции"),
            title="Корреляция Спирмана",
            width=700, height=600
        )
        fig_spearman.update_traces(text=spearman_sig, texttemplate="%{text}", textfont={"size": 14})
        st.plotly_chart(fig_spearman, use_container_width=True, key=f"spearman_corr_{uuid.uuid4()}")
    except Exception as e:
        st.error(f"Ошибка при построении карты Спирмана: {str(e)}")

    # Корреляционное отношение (η²)
    st.subheader("Корреляционное отношение (η²) для целевой переменной A (на исходных данных)")
    try:
        eta_squared = calculate_correlation_ratio(df, "A")
        eta_df = pd.DataFrame.from_dict(eta_squared, orient='index', columns=['η²']).round(3)
        eta_df = eta_df.loc[feature_cols]
        fig_eta = px.bar(
            eta_df,
            y='η²',
            color='η²',
            color_continuous_scale=px.colors.diverging.RdBu,
            range_color=[0, 1],
            labels={'index': 'Факторы', 'y': 'Корреляционное отношение η²'},
            text='η²',
            title="Сила нелинейной зависимости признаков от A",
            width=700, height=500
        )
        fig_eta.update_traces(texttemplate='%{text:.3f}', textposition='outside', textfont_size=14)
        st.plotly_chart(fig_eta, use_container_width=True, key=f"eta_squared_{uuid.uuid4()}")
    except Exception as e:
        st.error(f"Ошибка при расчете η²: {str(e)}")

    # VIF — на исходных данных
    X_vif = df[feature_cols]
    if X_vif.empty:
        st.warning("Нет признаков для анализа VIF")
        return df

    st.subheader("Анализ мультиколлинеарности (VIF) — на исходных данных")
    vif_data = calculate_vif(X_vif)
    vif_data = vif_data.set_index('feature').reindex(feature_cols).reset_index()
    fig_vif = px.bar(
        vif_data,
        x='feature',
        y='VIF',
        color='VIF',
        color_continuous_scale=['green', 'orange', 'red'],
        range_color=[0, 20],
        labels={'feature': 'Признак', 'y': 'VIF значение'},
        text='VIF',
        title="VIF — анализ мультиколлинеарности",
        width=700, height=500
    )
    fig_vif.add_hline(y=5, line_dash="dash", line_color="orange", annotation_text="Умеренная мультиколлинеарность")
    fig_vif.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Высокая мультиколлинеарность")
    fig_vif.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig_vif, use_container_width=True, key=f"vif_analysis_{uuid.uuid4()}")

    high_vif = vif_data[vif_data["VIF"] >= 10]
    if not high_vif.empty:
        st.warning("Обнаружены признаки с высокой мультиколлинеарностью (VIF ≥ 10):")
        st.dataframe(high_vif)
        st.info(
            "Вы можете удалить признаки с высоким VIF на вкладке **'Регрессионный анализ'** перед обучением моделей.")
    else:
        st.success("Значительной мультиколлинеарности не обнаружено (VIF < 10 для всех признаков)")

    return df

# Рассчитывает корреляционное отношение η² между признаками и целевой переменной.
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

# Вычисляет p-значения для матрицы корреляции (Пирсон или Спирман).
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

# Добавляет звездочки значимости (***, **, *) к коэффициентам корреляции на основе p-значений.
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

# Проводит статистический анализ модели (ANOVA) с помощью statsmodels, выводит p-значение F-статистики.
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

# Унифицированная функция предсказания с учетом масштабирования и нелинейных преобразований.
def predict_with_model(model, model_name, X_input, result):
    """
    Унифицированная функция предсказания.
    Теперь использует scaler из result, если он есть.
    """
    X_input = X_input.copy()
    scaling_method = result.get("scaling_method", "Нет")
    scaler = result.get("scaler", None)

    try:
        # Применяем масштабирование (кроме нейросети, у которой он встроен)
        if model_name != "Нейронная сеть" and scaling_method != "Нет" and scaler is not None:
            X_scaled = scaler.transform(X_input)
            X_input = pd.DataFrame(X_scaled, columns=X_input.columns)

        # Применяем нелинейные преобразования
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
        return float(pred)
    except Exception as e:
        st.error(f"Ошибка предсказания: {str(e)}")
        return np.nan

# Формирует и отображает формулу модели с коэффициентами, с учетом масштабирования и значимости.
def show_formula(coefficients, intercept, feature_names, regression_type, p_values=None, model_pipeline=None):
    # Инициализация
    scaler = None
    sigma = None
    mu = None

    # Определение типа скалера и его параметров
    if model_pipeline is not None:
        if 'standardscaler' in model_pipeline.named_steps:
            scaler = model_pipeline.named_steps['standardscaler']
            sigma = scaler.scale_
            mu = scaler.mean_
        elif 'minmaxscaler' in model_pipeline.named_steps:
            scaler = model_pipeline.named_steps['minmaxscaler']
            # Для MinMaxScaler: X_scaled = (X - X.min()) / (X.max() - X.min())
            sigma = scaler.data_max_ - scaler.data_min_  # диапазон (max - min)
            mu = scaler.data_min_  # минимальное значение
        elif 'robustscaler' in model_pipeline.named_steps:
            scaler = model_pipeline.named_steps['robustscaler']
            # Для RobustScaler: X_scaled = (X - median) / IQR
            sigma = scaler.scale_  # IQR (межквартильный размах)
            mu = scaler.center_  # медиана

    # Пересчет коэффициентов в исходный масштаб
    if scaler is not None and sigma is not None and mu is not None:
        try:
            beta_scaled = coefficients
            original_coefs = beta_scaled / sigma
            original_intercept = intercept - np.sum(beta_scaled * mu / sigma)
            coefficients = original_coefs
            intercept = original_intercept
            st.caption("Формула показана в **исходном масштабе** признаков (до масштабирования)")
        except Exception as e:
            st.warning(f"Ошибка при преобразовании коэффициентов: {str(e)}")
            st.caption("Формула показана в масштабированных данных")

    formula_parts = []
    if regression_type in ["Линейная", "Lasso"]:
        formula_parts.append(f"{intercept:.4f}")
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            significance = ""
            if p_values is not None and i < len(p_values):
                if p_values[i] < 0.001:
                    significance = "***"
                elif p_values[i] < 0.01:
                    significance = "**"
                elif p_values[i] < 0.05:
                    significance = "*"
            formula_parts.append(f"{coef:.4f}{significance}*{name}")
        formula = "A = " + " + ".join(formula_parts)
    elif regression_type in ["Квадратическая", "Кубическая"]:
        formula_parts.append(f"{intercept:.4f}")
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            if "^" in name:
                base, power = name.split("^")
                term = f"{coef:.4f}*{base}^{power}"
            else:
                term = f"{coef:.4f}*{name}"
            formula_parts.append(term)
        formula = "A = " + " + ".join(formula_parts)
    elif regression_type == "Логарифмическая":
        formula_parts.append(f"{intercept:.4f}")
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            significance = ""
            if p_values is not None and i < len(p_values):
                if p_values[i] < 0.001:
                    significance = "***"
                elif p_values[i] < 0.01:
                    significance = "**"
                elif p_values[i] < 0.05:
                    significance = "*"
            formula_parts.append(f"{coef:.4f}{significance}*log({name})")
        formula = "A = " + " + ".join(formula_parts)
    elif regression_type == "Экспоненциальная":
        a = np.exp(intercept)
        terms = []
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            significance = ""
            if p_values is not None and i < len(p_values):
                if p_values[i] < 0.001:
                    significance = "***"
                elif p_values[i] < 0.01:
                    significance = "**"
                elif p_values[i] < 0.05:
                    significance = "*"
            terms.append(f"{coef:.4f}{significance}*{name}")
        formula = f"A = {a:.4f} * exp(" + " + ".join(terms) + ")"
    elif regression_type == "Степенная":
        a = np.exp(intercept)
        terms = []
        for i, (coef, name) in enumerate(zip(coefficients, feature_names)):
            significance = ""
            if p_values is not None and i < len(p_values):
                if p_values[i] < 0.001:
                    significance = "***"
                elif p_values[i] < 0.01:
                    significance = "**"
                elif p_values[i] < 0.05:
                    significance = "*"
            terms.append(f"{name}^{coef:.4f}{significance}")
        formula = f"A = {a:.4f} * " + " * ".join(terms)
    else:
        formula = "Формула недоступна для выбранного типа регрессии"

    st.subheader("Формула модели")
    st.write(formula)
    if p_values is not None:
        st.markdown("""
        **Обозначения уровня значимости:**
        - \*** p < 0.001 — очень высокая значимость
        - \** p < 0.01 — высокая значимость
        - \* p < 0.05 — статистически значимо

        ⚠️ Пояснение:
        Звёздочки (*) у коэффициентов показывают, насколько признак статистически значимо влияет на целевую переменную.
        Чем меньше ( p )-значение, тем сильнее доказательства против гипотезы "коэффициент равен нулю".
        """)

# Визуализирует важность признаков (по коэффициентам или permutation importance).
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

# Строит график "фактические vs предсказанные" значения с идеальной линией.
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

# Строит график остатков и проверяет их нормальность с помощью теста Шапиро-Уилка.
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

# Извлекает объект PolynomialFeatures из pipeline модели.
def get_poly_features_from_pipeline(pipeline):
    for step_name, step in pipeline.named_steps.items():
        if isinstance(step, PolynomialFeatures):
            return step
    return None

# Строит 3D-поверхность отклика для двух выбранных признаков.
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

        # --- Фильтрация данных: оставить только близкие к фиксированным значениям ---
        tolerance = 0.1  # Доля от диапазона признака
        X_filtered = X.copy()
        y_filtered = y.copy()

        for feature, fixed_val in fixed_values.items():
            if feature in X_filtered.columns:
                min_val = X_filtered[feature].min()
                max_val = X_filtered[feature].max()
                range_val = max_val - min_val
                tol = tolerance * range_val if range_val != 0 else 0.1
                mask = (X_filtered[feature] >= fixed_val - tol) & (X_filtered[feature] <= fixed_val + tol)
                X_filtered = X_filtered[mask]
                y_filtered = y_filtered[mask]

        if len(X_filtered) == 0:
            st.info("Нет данных, близких к выбранным фиксированным значениям. Точки не отображаются.")
        else:
            st.info(f"На графике отображаются **{len(X_filtered)}** точек, близких к фиксированным значениям.")

        # --- Построение сетки для поверхности ---
        x_range = np.linspace(X[x_axis].min(), X[x_axis].max(), 20)
        y_range = np.linspace(X[y_axis].min(), X[y_axis].max(), 20)
        xx, yy = np.meshgrid(x_range, y_range)

        predict_data = pd.DataFrame({x_axis: xx.ravel(), y_axis: yy.ravel()})
        for feature, value in fixed_values.items():
            predict_data[feature] = value
        predict_data = predict_data[original_features]

        # Применение нелинейных преобразований (если нужно)
        if regression_type == "Логарифмическая":
            predict_data = np.log(predict_data.clip(lower=1e-9))
            predict_data = np.nan_to_num(predict_data, posinf=0, neginf=0)
        elif regression_type == "Степенная":
            predict_data = np.log(predict_data.clip(lower=1e-9))
            predict_data = np.nan_to_num(predict_data, posinf=0, neginf=0)

        # Предсказание
        try:
            if regression_type == "Экспоненциальная":
                zz = np.exp(model.predict(predict_data)).reshape(xx.shape)
            elif regression_type == "Степенная":
                zz = np.exp(model.predict(predict_data)).reshape(xx.shape)
            elif regression_type == "Логарифмическая":
                zz = model.predict(predict_data).reshape(xx.shape)
            else:
                zz = model.predict(predict_data).reshape(xx.shape)

            # --- 3D график ---
            fig = go.Figure()

            # Поверхность отклика
            fig.add_trace(go.Surface(
                x=xx, y=yy, z=zz,
                colorscale='Viridis',
                opacity=0.8,
                name='Поверхность отклика',
                showscale=True,
                colorbar=dict(title="Предсказание")
            ))

            # Исходные данные (только близкие)
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

# Предоставляет кнопку для скачивания обученной модели (в зависимости от типа).
def save_trained_model(model, model_name):
    try:
        if isinstance(model, Sequential):
            buffer = BytesIO()
            model.save(buffer, save_format='h5')
            buffer.seek(0)
            st.download_button(
                label=f"Скачать модель {model_name}",
                data=buffer,
                file_name=f"{model_name}_model.h5",
                mime="application/octet-stream",
                key=f"download_{model_name.replace(' ', '_')}_{uuid.uuid4()}"
            )
        else:
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
        st.error(f"Ошибка при сохранении модели: {str(e)}")

# Обучает одну модель заданного типа с кросс-валидацией и параметрами.
def train_model(reg_type, X_train, y_train, X_test, y_test, feature_cols, positive_mask_train, positive_mask_test,
                scaling_method):
    try:
        model = None
        coefficients = None
        intercept = None
        feature_names = feature_cols.copy()
        original_features = feature_cols.copy()
        cv = KFold(n_splits=5, shuffle=True, random_state=42)

        # Выбор скалера
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

        # --- Предсказания ---
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
                "r2": r2,
                "mse": mean_squared_error(y_test_vals, y_pred_vals),
                "rmse": np.sqrt(mean_squared_error(y_test_vals, y_pred_vals)),
                "mae": mean_absolute_error(y_test_vals, y_pred_vals),
                "mape": mean_absolute_percentage_error(y_test_vals, y_pred_vals)
            },
            "coefficients": coefficients,
            "intercept": intercept,
            "feature_names": feature_names,
            "original_features": original_features,
            "y_test": y_test_vals,
            "y_pred": y_pred_vals,
            "regression_type": reg_type,
            "color": color
        }

    except Exception as e:
        logger.error(f"Ошибка при обучении модели {reg_type}: {str(e)}")
        return {"error": str(e)}

# Запускает обучение всех моделей параллельно с прогресс-баром.
def run_all_regressions(df, scaling_method):
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
            if reg_type in ["Логарифмическая", "Экспоненциальная", "Степенная"] and not positive_data_available:
                results[reg_type] = {"error": "Требуются положительные значения"}
                progress_bar.progress((i + 1) / len(regression_types))
                status_text.text(f"Обработано {i + 1} из {len(regression_types)} моделей")
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
                progress_bar.progress((i + 1) / len(regression_types))
                status_text.text(f"Обработано {i + 1} из {len(regression_types)} моделей")
            except Exception as e:
                results[reg_type] = {"error": str(e)}
                logger.error(f"Ошибка при обучении модели {reg_type}: {str(e)}")

    return results, X_train, y_train

# Оценивает модель с помощью кросс-валидации по метрике R².
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

# Отображает сравнение моделей по метрикам и детальные результаты.
def show_regression_results(results, X_train, y_train):
    st.subheader("Сравнение моделей по R²")
    comparison_data = []
    for reg_type, res in results.items():
        if "error" not in res:
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
                    show_formula(res["coefficients"], res["intercept"], res["feature_names"], reg_type, p_values,
                                 model_pipeline=res["model"])
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

            # --- Кнопка сохранения модели (только для ML-моделей) ---
            st.markdown("---")
            # Список моделей, которые являются аналитическими (не ML)
            analytical_models = [
                "Линейная", "Квадратическая", "Кубическая",
                "Логарифмическая", "Экспоненциальная", "Степенная", "Lasso"
            ]

            if reg_type not in analytical_models:
                save_trained_model(res["model"], reg_type)
            else:
                st.info("Аналитические модели представлены формулой. Сохранение модели не требуется.")

# Анализирует выбросы в данных по методу IQR (только диагностика).
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
        # 🔴 УБРАНО: st.info("Рекомендуется обработать выбросы вручную или выбрать метод ниже.")
    else:
        st.success("Выбросы не обнаружены (по методу IQR).")

    return df

# Отображает панель статуса на боковой панели.
def show_status_panel():
    st.sidebar.header("📊 Статус анализа")

    def status_icon(condition):
        return "✅" if condition else "🟡"

    st.sidebar.write(f"{status_icon(st.session_state.status['data_loaded'])} **Данные загружены**")
    st.sidebar.write(f"{status_icon(st.session_state.status['outliers_detected'])} **Выбросы обнаружены**")
    st.sidebar.write(f"{status_icon(st.session_state.status['outliers_handled'])} **Выбросы обработаны**")
    st.sidebar.write(f"{status_icon(st.session_state.status['vif_analyzed'])} **VIF проанализирован**")
    st.sidebar.write(f"{status_icon(st.session_state.status['models_trained'])} **Модели обучены**")

    # Прогресс
    progress = sum(st.session_state.status.values()) / len(st.session_state.status)
    st.sidebar.progress(progress)
    st.sidebar.caption(f"Готово: {int(progress * 100)}%")

# Основная функция приложения Streamlit.
def main():
    st.title("Полиномиальная регрессия и машинное обучение")
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'processed_df' not in st.session_state:
        st.session_state.processed_df = None
    if 'results' not in st.session_state:
        st.session_state.results = None

    # Инициализация состояний
    if 'status' not in st.session_state:
        st.session_state.status = {
            'data_loaded': False,
            'outliers_detected': False,
            'outliers_handled': False,
            'vif_analyzed': False,
            'models_trained': False
        }
    show_status_panel()

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "1. Загрузка данных",
        "2. Анализ данных",
        "3. Корреляционный анализ",
        "4. Анализ масштабирования",  # <-- НОВЫЙ ТАБ
        "5. Регрессионный анализ",
        "6. Оптимизация"
    ])

    with tab1:
        st.header("Загрузка и проверка данных")
        df = load_data()
        if df is not None:
            st.session_state.df = df.copy()
            st.session_state.processed_df = df.copy()
            st.session_state.status['data_loaded'] = True  # <-- ДОБАВЛЕНО
            st.success(f"Данные успешно загружены! Количество строк: {len(df)}, столбцов: {len(df.columns)}")
        else:
            st.session_state.df = None
            st.session_state.processed_df = None
            st.session_state.status['data_loaded'] = False  # Явное обновление

    with tab2:
        st.header("Анализ исходных данных")
        if st.session_state.df is not None:
            show_descriptive_analysis(st.session_state.df)
            st.session_state.processed_df = data_preparation(st.session_state.df)
            # Не показываем данные здесь — обработка ещё не применена
            st.info("Данные будут обработаны на вкладке 'Корреляционный анализ' после выбора метода.")
        else:
            st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'")

    with tab3:
        st.header("Корреляционный анализ")
        # 🔒 Гарантированная проверка: загружены ли данные?
        if 'df' not in st.session_state or st.session_state.df is None:
            st.warning("Пожалуйста, загрузите данные на вкладке 'Загрузка данных'.")
            st.session_state.status['outliers_detected'] = False
            st.session_state.status['outliers_handled'] = False
            st.session_state.status['vif_analyzed'] = False
            return
        # Всегда работаем с копией исходных данных
        df_original = st.session_state.df.copy()
        numeric_cols = df_original.select_dtypes(include=[np.number]).columns
        target_col = 'A'
        if target_col not in df_original.columns:
            st.error("Отсутствует целевая переменная 'A'")
            return

        # --- Анализ выбросов (только диагностика) ---
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
            st.session_state.status['outliers_detected'] = True
        else:
            st.success("Выбросы не обнаружены (по методу IQR).")
            st.session_state.status['outliers_detected'] = True
            st.session_state.status['outliers_handled'] = True

        # --- Выбор метода обработки ---
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
            st.session_state.status['outliers_handled'] = True
        else:
            st.session_state.status['outliers_handled'] = False

        st.session_state.processed_df = df_processed.copy()

        # --- Сравнительный анализ: до и после ---
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

        # --- Функция для добавления звёздочек ---
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

        # --- Пирсон ---
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

        # --- Спирман ---
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

        # --- η² ---
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

        # --- VIF ---
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### 📊 VIF (исходные)")
            if len(feature_cols) > 0:
                vif_orig = calculate_vif(df_original[feature_cols])
                fig5 = px.bar(vif_orig, x='feature', y='VIF', color='VIF',
                              color_continuous_scale=['green', 'orange', 'red'], range_color=[0, 20],
                              text='VIF', title="VIF (до)")
                fig5.add_hline(y=10, line_dash="dash", line_color="red")
                st.plotly_chart(fig5, use_container_width=True)
            else:
                st.info("Нет признаков для VIF.")

        with col2:
            st.markdown("### 📈 VIF (обработанные)")
            if len(feature_cols) > 0:
                vif_proc = calculate_vif(df_processed[feature_cols])
                fig6 = px.bar(vif_proc, x='feature', y='VIF', color='VIF',
                              color_continuous_scale=['green', 'orange', 'red'], range_color=[0, 20],
                              text='VIF', title="VIF (после)")
                fig6.add_hline(y=10, line_dash="dash", line_color="red")
                st.plotly_chart(fig6, use_container_width=True)
            else:
                st.info("Нет признаков для VIF.")

        # --- Анализ изменений (дельта) ---
        st.subheader("📉 Анализ изменений (после - до)")

        # Δ Пирсон
        delta_corr = corr_proc - corr_orig
        st.markdown("### 🔄 Δ Корреляция Пирсона (после - до)")
        fig_dc = px.imshow(delta_corr, text_auto=True, color_continuous_scale='RdBu', zmin=-0.5, zmax=0.5,
                           title="Δ Пирсон")
        st.plotly_chart(fig_dc, use_container_width=True)

        # Δ Спирман
        delta_corr_s = corr_proc_s - corr_orig_s
        st.markdown("### 🔄 Δ Корреляция Спирмана (после - до)")
        fig_ds = px.imshow(delta_corr_s, text_auto=True, color_continuous_scale='RdBu', zmin=-0.5, zmax=0.5,
                           title="Δ Спирман")
        st.plotly_chart(fig_ds, use_container_width=True)

        # Δ η²
        eta_delta = {f: round(eta_proc.get(f, 0) - eta_orig.get(f, 0), 3) for f in feature_cols}
        eta_delta_df = pd.DataFrame.from_dict(eta_delta, orient='index', columns=['Δ η²']).round(3)
        st.markdown("### 🔗 Δ η² (после - до)")
        fig_de = px.bar(eta_delta_df, y='Δ η²', color='Δ η²',
                        color_continuous_scale=['red', 'white', 'green'], range_color=[-0.5, 0.5],
                        text='Δ η²', title="Δ η²")
        fig_de.add_hline(y=0, line_dash="dash", line_color="black")
        fig_de.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_de, use_container_width=True)

        # Δ VIF
        vif_delta = vif_proc.set_index('feature').subtract(vif_orig.set_index('feature'), fill_value=0).round(
            3).reset_index()
        st.markdown("### 📉 Δ VIF (после - до)")
        fig_dv = px.bar(vif_delta, x='feature', y='VIF', color='VIF',
                        color_continuous_scale=['red', 'white', 'green'], range_color=[-10, 10],
                        text='VIF', title="Δ VIF")
        fig_dv.add_hline(y=0, line_dash="dash", line_color="black")
        fig_dv.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig_dv, use_container_width=True)

        # Финальный статус
        st.session_state.status['vif_analyzed'] = True

    with tab4:
        st.header("Анализ масштабирования данных")

        if st.session_state.processed_df is None:
            st.warning("Пожалуйста, выполните корреляционный анализ на предыдущей вкладке.")
        else:
            df = st.session_state.processed_df.copy()
            feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]

            if len(feature_cols) == 0:
                st.warning("Нет числовых признаков для масштабирования.")
            else:
                # Подготовка данных
                X = df[feature_cols]

                # Методы масштабирования
                scalers = {
                    "StandardScaler (Z-score)": StandardScaler(),
                    "MinMaxScaler ([0,1])": MinMaxScaler(),
                    "RobustScaler (медиана/IQR)": RobustScaler(),
                    "Без масштабирования": None
                }

                st.subheader("Влияние масштабирования на статистику признаков")

                # Собираем статистику
                stats = []
                for name, scaler in scalers.items():
                    if scaler is None:
                        X_scaled = X.copy()
                    else:
                        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

                    for col in feature_cols:
                        row = {
                            "Метод": name,
                            "Признак": col,
                            "Среднее": X_scaled[col].mean(),
                            "Std": X_scaled[col].std() if scaler is not None else X[col].std(),
                            "Min": X_scaled[col].min(),
                            "Max": X_scaled[col].max(),
                            "IQR": X_scaled[col].quantile(0.75) - X_scaled[col].quantile(0.25)
                        }
                        stats.append(row)

                stats_df = pd.DataFrame(stats)

                # Группируем по методу
                method_stats = stats_df.groupby("Метод").agg(
                    Среднее_средних=("Среднее", "mean"),
                    Среднее_std=("Std", "mean"),
                    Средний_IQR=("IQR", "mean")
                ).round(3)

                st.dataframe(method_stats)

                # --- Влияние на VIF ---
                st.subheader("Влияние на мультиколлинеарность (VIF)")

                st.markdown("""
                **📌 Влияние масштабирования на VIF**:
                - Масштабирование **не устраняет** мультиколлинеарность, так как VIF зависит от корреляций между признаками, а не от их масштаба.
                - Однако признаки с разным масштабом могут искажать интерпретацию коэффициентов в линейных моделях.
                - `RobustScaler` и `StandardScaler` помогают стабилизировать обучение, но не влияют на VIF напрямую.
                """)

                col1, col2, col3, col4 = st.columns(4)
                for i, (name, scaler) in enumerate(scalers.items()):
                    with [col1, col2, col3, col4][i]:
                        if scaler is None:
                            X_scaled = X.copy()
                        else:
                            X_scaled = scaler.fit_transform(X)
                            X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

                        vif_data = calculate_vif(X_scaled)
                        avg_vif = vif_data['VIF'].mean()
                        max_vif = vif_data['VIF'].max()

                        st.markdown(f"### {name}")
                        st.metric("Средний VIF", f"{avg_vif:.2f}")
                        st.metric("Макс. VIF", f"{max_vif:.2f}")

                # --- Влияние на выбросы ---
                st.subheader("Влияние на выбросы (IQR)")
                outlier_summary = []
                for name, scaler in scalers.items():
                    if scaler is None:
                        X_scaled = X.copy()
                    else:
                        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)

                    total_outliers = 0
                    for col in feature_cols:
                        Q1 = X_scaled[col].quantile(0.25)
                        Q3 = X_scaled[col].quantile(0.75)
                        IQR = Q3 - Q1
                        lower = Q1 - 1.5 * IQR
                        upper = Q3 + 1.5 * IQR
                        total_outliers += ((X_scaled[col] < lower) | (X_scaled[col] > upper)).sum()

                    outlier_summary.append({"Метод": name, "Выбросов": total_outliers})

                outlier_df = pd.DataFrame(outlier_summary)
                fig_outliers = px.bar(outlier_df, x="Метод", y="Выбросов", color="Выбросов",
                                      title="Количество выбросов после масштабирования")
                st.plotly_chart(fig_outliers, use_container_width=True)

                # --- Рекомендация ---
                st.subheader("🎯 Рекомендация по методу масштабирования")

                # Логика рекомендации
                robust_outliers = outlier_df.set_index("Метод").loc["RobustScaler (медиана/IQR)", "Выбросов"]
                std_outliers = outlier_df.set_index("Метод").loc["StandardScaler (Z-score)", "Выбросов"]
                minmax_outliers = outlier_df.set_index("Метод").loc["MinMaxScaler ([0,1])", "Выбросов"]

                robust_vif = vif_data[vif_data['feature'] == feature_cols[0]].iloc[0]['VIF']  # пример
                high_vif = (vif_data['VIF'] > 10).sum() > 0

                if robust_outliers < minmax_outliers and robust_outliers < std_outliers:
                    st.success("✅ **Рекомендовано: `RobustScaler`**")
                    st.write("• Наилучшее подавление влияния выбросов.")
                    st.write("• Устойчив к аномалиям.")
                elif minmax_outliers <= std_outliers:
                    st.info("🟡 **Рекомендовано: `MinMaxScaler`**")
                    st.write("• Подходит для нейросетей и моделей, чувствительных к диапазону.")
                    st.write("• Хорошо, если выбросы обработаны.")
                else:
                    st.info("🟡 **Рекомендовано: `StandardScaler`**")
                    st.write("• Классический выбор для линейных моделей.")
                    st.write("• Работает, если выбросы не критичны.")

                # Дополнительные советы
                if high_vif:
                    st.warning(
                        "⚠️ Обнаружена высокая мультиколлинеарность. Рассмотрите удаление признаков в следующем табе.")

                st.caption("💡 Совет: Выбор можно будет подтвердить или изменить в табе 'Регрессионный анализ'.")

                # Сохраняем для следующего таба
                # Определяем рекомендованный метод по полному имени
                if robust_outliers < minmax_outliers and robust_outliers < std_outliers:
                    recommended_full = "RobustScaler (устойчивый)"
                elif minmax_outliers <= std_outliers:
                    recommended_full = "MinMaxScaler (нормализация)"
                else:
                    recommended_full = "StandardScaler (стандартизация)"

                # Сохраняем полное имя для использования в selectbox
                st.session_state.scaler_recommendation = {
                    "recommended": recommended_full
                }

    with tab5:
        st.header("Регрессионный анализ")
        if st.session_state.processed_df is not None:
            if "A" not in st.session_state.processed_df.columns:
                st.error("Отсутствует целевая переменная 'A'")
                st.session_state.status['models_trained'] = False
            else:
                df = st.session_state.processed_df.copy()
                feature_cols = [col for col in ['B', 'C', 'D', 'E', 'F', 'G', 'H'] if col in df.columns]

                # Инициализация списка удаления признаков
                if 'vif_remove_list' not in st.session_state:
                    st.session_state.vif_remove_list = []

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

                # Выбор метода масштабирования
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

                # Кнопка запуска
                if st.button("Запустить все регрессии", key="run_regressions"):
                    with st.spinner("Подготовка данных и обучение моделей..."):
                        # Применяем удаление признаков
                        if st.session_state.vif_remove_list:
                            available_to_remove = [f for f in st.session_state.vif_remove_list if f in df.columns]
                            if available_to_remove:
                                df = df.drop(columns=available_to_remove)
                                st.success(f"Удалены признаки: {', '.join(available_to_remove)}")
                            else:
                                st.info("Признаки с высоким VIF не выбраны для удаления.")
                        else:
                            st.info("Признаки с высоким VIF не помечены для удаления.")

                        # Сохраняем обновлённый df
                        st.session_state.processed_df = df.copy()

                        # Запускаем регрессии
                        results, X_train, y_train = run_all_regressions(df, scaling_method)

                        # Добавляем информацию о масштабировании в каждый результат
                        for model_name in results:
                            if "error" not in results[model_name]:  # Только успешные модели
                                # Сохраняем метод масштабирования
                                results[model_name]["scaling_method"] = scaling_method

                                # Сохраняем объект scaler, если он есть (для моделей, кроме нейросети)
                                # Нейросеть имеет свой встроенный scaler, его не нужно дублировать
                                if model_name != "Нейронная сеть" and scaling_method != "Нет":
                                    # Предполагается, что run_all_regressions возвращает scaler в X_train или отдельно
                                    # Если scaler не возвращается, его нужно создать и сохранить здесь
                                    if 'scaler' in results[model_name]:
                                        # Уже есть (если run_all_regressions его возвращает)
                                        pass
                                    else:
                                        # Создаём и обучаем scaler здесь, если его нет
                                        scaler_obj = get_scaler_from_name(scaling_method)
                                        feature_cols = results[model_name]["original_features"]
                                        X_for_scaling = df[feature_cols]
                                        results[model_name]["scaler"] = scaler_obj.fit(X_for_scaling)

                        # Сохраняем в сессию
                        st.session_state.results = results
                        st.session_state.X_train = X_train
                        st.session_state.y_train = y_train
                        st.session_state.status['models_trained'] = True
                        st.rerun()

                # Показ результатов
                if st.session_state.results is not None:
                    show_regression_results(st.session_state.results, st.session_state.X_train,
                                            st.session_state.y_train)
                else:
                    st.session_state.status['models_trained'] = False
                    st.info("Нажмите 'Запустить все регрессии', чтобы обучить модели.")

        else:
            st.warning("Пожалуйста, загрузите и обработайте данные на предыдущих вкладках.")
            st.session_state.status['models_trained'] = False

    with tab6:
        st.header("Оптимизация и симуляция")
        # Инициализация session_state для сохранения сессии
        if 'simulator_inputs' not in st.session_state:
            st.session_state.simulator_inputs = {}
        if 'optimization_history' not in st.session_state:
            st.session_state.optimization_history = []
        if 'optimization_result' not in st.session_state:
            st.session_state.optimization_result = None

        if st.session_state.results is None or st.session_state.processed_df is None:
            st.warning("Пожалуйста, выполните регрессионный анализ на вкладке 'Регрессионный анализ'.")
        else:
            valid_results = {k: v for k, v in st.session_state.results.items() if "error" not in v}
            if not valid_results:
                st.warning("Нет обученных моделей для оптимизации.")
            else:
                # --- Выбор модели ---
                model_options = list(valid_results.keys())
                if 'selected_opt_model' not in st.session_state:
                    st.session_state.selected_opt_model = model_options[0]
                st.session_state.selected_opt_model = st.selectbox(
                    "Выберите модель для оптимизации",
                    model_options,
                    index=model_options.index(st.session_state.selected_opt_model)
                )
                best_model_name = st.session_state.selected_opt_model
                best_result = valid_results[best_model_name]
                model = best_result["model"]
                feature_cols = best_result["original_features"]
                df = st.session_state.processed_df
                st.success(f"Используется модель: **{best_model_name}** (R² = {best_result['metrics']['r2']:.4f})")

                # --- Режим: Симулятор ---
                st.subheader("🎮 Интерактивный симулятор")
                st.caption(
                    "Значения признаков ограничены диапазоном, представленным в обучающих данных. "
                    "Это предотвращает некорректные предсказания за пределами обученного диапазона."
                )
                cols = st.columns(len(feature_cols))
                inputs = {}
                for i, col in enumerate(feature_cols):
                    with cols[i % len(cols)]:
                        # Восстановление значений из сессии
                        if col not in st.session_state.simulator_inputs:
                            st.session_state.simulator_inputs[col] = float(df[col].median())
                        value = st.number_input(
                            f"{col}",
                            value=st.session_state.simulator_inputs[col],
                            min_value=float(df[col].min()),
                            max_value=float(df[col].max()),
                            step=0.01,
                            help=f"Допустимый диапазон: от {df[col].min():.3f} до {df[col].max():.3f} (на основе обучающих данных)",
                            key=f"sim_input_{col}"
                        )
                        st.session_state.simulator_inputs[col] = value
                        inputs[col] = value

                X_input = pd.DataFrame([inputs])
                try:
                    pred = predict_with_model(model, best_model_name, X_input, best_result)
                    st.metric("Предсказанное значение A", f"{pred:.4f}")
                except Exception as e:
                    st.error(f"Ошибка предсказания: {str(e)}")

                st.markdown("---")

                # --- Режим: Оптимизация ---
                st.subheader("⚙️ Оптимизация (scipy.optimize)")
                optimization_mode = st.radio(
                    "Цель оптимизации",
                    ["Максимизировать A", "Минимизировать A", "Достичь целевого значения"],
                    key="opt_mode"
                )
                target_value = None
                if optimization_mode == "Достичь целевого значения":
                    target_value = st.number_input(
                        "Целевое значение A",
                        value=float(df["A"].mean()),
                        key="target_value_input"
                    )

                # Диапазоны
                st.write("### Диапазоны факторов")
                bounds = []
                cols = st.columns(len(feature_cols))
                for i, col in enumerate(feature_cols):
                    with cols[i % len(cols)]:
                        low_key = f"bound_low_{col}"
                        high_key = f"bound_high_{col}"
                        if low_key not in st.session_state:
                            st.session_state[low_key] = float(df[col].min())
                        if high_key not in st.session_state:
                            st.session_state[high_key] = float(df[col].max())
                        low = st.number_input(f"Мин {col}", value=st.session_state[low_key], step=0.01, key=low_key)
                        high = st.number_input(f"Макс {col}", value=st.session_state[high_key], step=0.01, key=high_key)
                        bounds.append((low, high))

                # Кнопка запуска
                if st.button("Запустить оптимизацию", key="optimize_scipy"):
                    from scipy.optimize import minimize
                    # История вызовов
                    history = []

                    def predict_func(x):
                        x_clipped = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])
                        X_point = pd.DataFrame([x_clipped], columns=feature_cols)
                        pred = predict_with_model(model, best_model_name, X_point, best_result)
                        history.append((x_clipped.copy(), pred))
                        return pred

                    def obj_func(x):
                        pred = predict_func(x)
                        if optimization_mode == "Максимизировать A":
                            return -pred
                        elif optimization_mode == "Минимизировать A":
                            return pred
                        else:
                            return (pred - target_value) ** 2

                    # Запуск
                    result = minimize(
                        obj_func,
                        x0=[df[col].median() for col in feature_cols],
                        bounds=bounds,
                        method='L-BFGS-B'
                    )

                    # Сохранение в сессию
                    st.session_state.optimization_history = history
                    st.session_state.optimization_result = {
                        "success": result.success,
                        "message": result.message,
                        "optimal_x": result.x,
                        "optimal_y": predict_func(result.x),
                        "mode": optimization_mode,
                        "target": target_value
                    }

                # Отображение результата
                if st.session_state.optimization_result:
                    res = st.session_state.optimization_result
                    if res["success"]:
                        st.success("✅ Оптимизация завершена успешно!")
                        if res["mode"] == "Достичь целевого значения":
                            st.write(f"**Целевое значение:** {res['target']}")
                            st.write(f"**Достигнутое A:** {res['optimal_y']:.4f}")
                            st.write(f"**Ошибка:** {abs(res['optimal_y'] - res['target']):.4f}")
                        else:
                            st.write(f"**Оптимальное A:** {res['optimal_y']:.4f}")
                        st.write("Рекомендуемые значения:")
                        for col, val in zip(feature_cols, res["optimal_x"]):
                            st.write(f"- **{col}:** {val:.4f}")
                    else:
                        st.error(f"Оптимизация не удалась: {res['message']}")

                # Визуализация траектории
                if st.session_state.optimization_history:
                    st.subheader("📊 Траектория оптимизации")
                    history_df = pd.DataFrame(
                        [dict(zip(feature_cols, x)) for x, y in st.session_state.optimization_history],
                        index=range(len(st.session_state.optimization_history))
                    )
                    history_df["A_pred"] = [y for x, y in st.session_state.optimization_history]
                    history_df["step"] = history_df.index

                    # График изменения A
                    fig_a = px.line(
                        history_df,
                        x="step",
                        y="A_pred",
                        title="Изменение предсказанного A на шагах оптимизации",
                        labels={"step": "Шаг", "A_pred": "Предсказанное A"}
                    )
                    st.plotly_chart(fig_a, use_container_width=True)

                    # Если 2+ фактора — можно показать первые два
                    if len(feature_cols) >= 2:
                        fig_traj = px.scatter(
                            history_df,
                            x=feature_cols[0],
                            y=feature_cols[1],
                            size="A_pred",
                            color="A_pred",
                            hover_data=["step"],
                            title=f"Траектория оптимизации: {feature_cols[0]} vs {feature_cols[1]}",
                            labels={feature_cols[0]: feature_cols[0], feature_cols[1]: feature_cols[1]}
                        )
                        fig_traj.update_traces(marker=dict(sizemode='diameter', sizeref=0.1))
                        st.plotly_chart(fig_traj, use_container_width=True)

                    # Кнопка сброса
                    if st.button("Очистить историю"):
                        st.session_state.optimization_history = []
                        st.session_state.optimization_result = None
                        st.rerun()


if __name__ == "__main__":
    main()
