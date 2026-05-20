"""
utils.py — вспомогательные функции для проекта covid19-xai-forecasting
Автор: Кибешев Д.М., ИТМО, 2025–2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ══════════════════════════════════════════════════════════════════
# 1. СТИЛЬ ГРАФИКОВ
# ══════════════════════════════════════════════════════════════════

def set_plot_style():
    """Единый стиль для всех графиков проекта."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.grid":         True,
        "grid.alpha":        0.3,
        "grid.linestyle":    "--",
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "legend.fontsize":   10,
        "xtick.labelsize":   10,
        "ytick.labelsize":   10,
    })


# ══════════════════════════════════════════════════════════════════
# 2. ПРЕПРОЦЕССИНГ ВРЕМЕННОГО РЯДА
# ══════════════════════════════════════════════════════════════════

def smooth_series(series: pd.Series, window: int = 7) -> pd.Series:
    """Скользящее среднее с заполнением краёв."""
    return series.rolling(window=window, min_periods=1, center=False).mean()


def mark_omicron(df: pd.DataFrame,
                 date_col: str = "date",
                 start: str = "2022-01-01") -> pd.DataFrame:
    """Добавляет бинарный флаг омикрон-волны."""
    df = df.copy()
    df["omicron_wave"] = (df[date_col] >= pd.Timestamp(start)).astype(int)
    return df


def fill_missing(df: pd.DataFrame, method: str = "linear") -> pd.DataFrame:
    """Линейная интерполяция пропусков по всем числовым столбцам."""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].interpolate(method=method, limit_direction="both")
    return df


def shift_features(df: pd.DataFrame,
                   feature_cols: list,
                   shift: int = 4) -> pd.DataFrame:
    """
    Сдвигает внешние признаки вперёд на shift дней.
    Логика: если сегодня День 0, то признак влияет через shift дней
    (инкубационный период COVID ≈ 4–5 дней).
    """
    df = df.copy()
    for col in feature_cols:
        df[col] = df[col].shift(shift)
    return df


def add_lag_features(df: pd.DataFrame,
                     target_col: str,
                     lags: range = range(1, 15)) -> pd.DataFrame:
    """Добавляет лаги целевой переменной (для univariate baseline)."""
    df = df.copy()
    for lag in lags:
        df[f"{target_col}_lag{lag}"] = df[target_col].shift(lag)
    return df


# ══════════════════════════════════════════════════════════════════
# 3. РАЗБИВКА НА TRAIN / TEST
# ══════════════════════════════════════════════════════════════════

def train_test_split_temporal(df: pd.DataFrame,
                               test_days: int = 31) -> tuple:
    """
    Разбивка временного ряда: последние test_days → test, остальное → train.
    Никакого случайного перемешивания — строго по времени.
    """
    train = df.iloc[:-test_days].copy()
    test  = df.iloc[-test_days:].copy()
    return train, test


def reduce_train_size(X_train: np.ndarray,
                      y_train: np.ndarray,
                      n: int) -> tuple:
    """Берёт последние n наблюдений обучающей выборки."""
    return X_train[-n:], y_train[-n:]


# ══════════════════════════════════════════════════════════════════
# 4. МЕТРИКИ
# ══════════════════════════════════════════════════════════════════

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    model_name: str = "Model") -> pd.DataFrame:
    """
    Вычисляет RMSE, MAE, MAPE, R² и возвращает DataFrame-строку.
    Используется для сравнения моделей.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE: избегаем деления на ноль
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    r2   = r2_score(y_true, y_pred)

    return pd.DataFrame({
        "Model": [model_name],
        "RMSE":  [round(rmse, 2)],
        "MAE":   [round(mae, 2)],
        "MAPE":  [round(mape, 2)],
        "R2":    [round(r2, 4)],
    })


def metrics_table(metrics_list: list) -> pd.DataFrame:
    """Объединяет список DataFrame-строк в итоговую таблицу."""
    return pd.concat(metrics_list, ignore_index=True)


# ══════════════════════════════════════════════════════════════════
# 5. ВИЗУАЛИЗАЦИИ
# ══════════════════════════════════════════════════════════════════

def plot_forecast(dates, y_true, y_pred, model_name: str,
                  save_path: str = None):
    """
    График: факт vs прогноз для тестового периода.
    """
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates, y_true, label="Факт", color="#2c3e50", linewidth=2)
    ax.plot(dates, y_pred, label=f"Прогноз ({model_name})",
            color="#e74c3c", linewidth=1.8, linestyle="--")
    ax.fill_between(dates, y_true, y_pred, alpha=0.1, color="#e74c3c")
    ax.set_ylabel("Новые случаи (сглаженные)")
    ax.set_title(f"Прогноз vs Факт — {model_name}")
    ax.legend()
    try:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    except Exception:
        pass
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_metrics_bar(metrics_df: pd.DataFrame,
                     metric: str = "RMSE",
                     save_path: str = None):
    """
    Столбчатая диаграмма сравнения моделей по одной метрике.
    """
    colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(metrics_df["Model"], metrics_df[metric],
                  color=colors[:len(metrics_df)], edgecolor="white", width=0.6)
    for bar, val in zip(bars, metrics_df[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + metrics_df[metric].max() * 0.01,
                f"{val:.1f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylabel(metric)
    ax.set_title(f"Сравнение моделей — {metric}")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_train_reduction(sizes: list, rmse_scores: list,
                          save_path: str = None):
    """
    График зависимости RMSE от размера обучающей выборки.
    Помогает найти оптимальный размер (Train Data Reduction).
    """
    best_n = sizes[np.argmin(rmse_scores)]
    best_rmse = min(rmse_scores)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(sizes, rmse_scores, color="#2980b9", linewidth=2, marker="o",
            markersize=4, markevery=5)
    ax.axvline(best_n, color="#e74c3c", linestyle="--", linewidth=1.5,
               label=f"Оптимум: {best_n} дней (RMSE={best_rmse:.1f})")
    ax.scatter([best_n], [best_rmse], color="#e74c3c", s=100, zorder=5)
    ax.set_xlabel("Размер обучающей выборки (дней)")
    ax.set_ylabel("RMSE на тестовой выборке")
    ax.set_title("Train Data Reduction: поиск оптимального размера выборки")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_comparison(comparison_df: pd.DataFrame,
                          save_path: str = None):
    """
    Тепловая карта рангов признаков по трём методам XAI.
    comparison_df должен содержать колонки: feature, rank_shap, rank_perm, rank_lime
    """
    import seaborn as sns
    rank_matrix = comparison_df.set_index("feature")[
        ["rank_shap", "rank_perm", "rank_lime"]
    ]
    rank_matrix.columns = ["TreeSHAP", "Permutation", "LIME"]

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(rank_matrix, annot=True, fmt=".0f", cmap="YlOrRd_r",
                ax=ax, linewidths=0.5,
                cbar_kws={"label": "Ранг (1 = важнее)"})
    ax.set_title("Сравнение методов XAI: ранги признаков")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
