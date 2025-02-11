import os
import warnings
warnings.filterwarnings("ignore")

from scipy.stats.mstats import winsorize
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from scipy.stats import norm

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture

from RCAEval.io.time_series import (
    convert_mem_mb,
    drop_constant,
    drop_extra,
    drop_near_constant,
    drop_time,
    preprocess,
    select_useful_cols,
)

# 1. Adaptive Z-Score
def adaptive_z_score(data, confidence=0.99):
    mean, std = np.mean(data), np.std(data)
    threshold = norm.ppf(confidence)
    z_scores = (data - mean) / std
    return data[np.abs(z_scores) <= threshold]

# 2. Isolation Forest
def isolation_forest_outlier(data, contamination=0.01):
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(data.reshape(-1, 1))
    return data[preds == 1]

# 3. Local Outlier Factor (LOF)
def local_outlier_factor(data, n_neighbors=20, contamination=0.01):
    model = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    preds = model.fit_predict(data.reshape(-1, 1))
    return data[preds == 1]

# 4. Interquartile Range (IQR)
def iqr_outlier(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# 5. Tukey’s Fences (IQR Alternative)
def tukey_outlier(data, k=2.2):  # More aggressive than IQR
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    return data[(data >= lower_bound) & (data <= upper_bound)]

# 6. DBSCAN Outlier Detection
def dbscan_outlier(data, eps=1.5, min_samples=5):
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data.reshape(-1, 1))
    return data[labels != -1]

# 7. Gaussian Mixture Model (GMM)
def gmm_outlier(data, n_components=2):
    model = GaussianMixture(n_components=n_components)
    model.fit(data.reshape(-1, 1))
    scores = model.score_samples(data.reshape(-1, 1))
    threshold = np.percentile(scores, 5)
    return data[scores > threshold]

# 8. Bayesian Outlier Detection (BOCD)
def bayesian_outlier(data):
    try:
        from bocd import BOCD
        model = BOCD()
        scores = model.fit_predict(data)
        return data[scores < 0.99]
    except ImportError:
        warnings.warn("BOCD package not found, skipping Bayesian Outlier Detection.")
        return data

# 9. Clipping Outliers
def clip_outlier(data, lower_percentile=5, upper_percentile=95):
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)

# 10. Winsorization
def winsorize_outlier(data, limits=(0.02, 0.02)):  # Trims top and bottom 5% of values
    return winsorize(data, limits=limits)

def baro(
    data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, outlier_method='adaptive_z', **kwargs
):
    if anomalies is None:
        normal_df = data[data["time"] < inject_time]
        anomal_df = data[data["time"] >= inject_time]
    else:
        normal_df = data.head(anomalies[0])
        anomal_df = data.tail(len(data) - anomalies[0])


    normal_df = preprocess(
        data=normal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    anomal_df = preprocess(
        data=anomal_df, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False)
    )

    # intersect
    intersects = [x for x in normal_df.columns if x in anomal_df.columns]
    normal_df = normal_df[intersects]
    anomal_df = anomal_df[intersects]

    ranks = []

    for col in normal_df.columns:
        a = normal_df[col].to_numpy()
        b = anomal_df[col].to_numpy()

        # Select and apply the chosen outlier detection method
        outlier_methods = {
            'adaptive_z': adaptive_z_score,
            'isolation_forest': isolation_forest_outlier,
            'lof': local_outlier_factor,
            'iqr': iqr_outlier,
            'tukey': tukey_outlier,
            'dbscan': dbscan_outlier,
            'gmm': gmm_outlier,
            'bayesian': bayesian_outlier,
            'winsorization': winsorize_outlier,
            'clipping': clip_outlier
        }

        if outlier_method in outlier_methods:
            a = outlier_methods[outlier_method](a)
            b = outlier_methods[outlier_method](b)

        scaler = RobustScaler().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    ranks = [x[0] for x in ranks]

    return {
        "node_names": normal_df.columns.to_list(),
        "ranks": ranks,
    }
    
    
def mmnsigma(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    scaler_function = kwargs.get("scaler_function", StandardScaler) 

    metric = data["metric"]
    logs = data["logs"]
    logts = data["logts"]
    traces = data["traces"]
    traces_err = data["tracets_err"]
    traces_lat = data["tracets_lat"]
    cluster_info = data["cluster_info"]
    
    # ==== PREPARE DATA ====
    # the metric is currently sampled for 1 seconds, resample for 15s by just take 1 point every 15 points
    metric = metric.iloc[::15, :]

    # == metric ==
    normal_metric = metric[metric["time"] < inject_time]
    anomal_metric = metric[metric["time"] >= inject_time]
    normal_metric = preprocess(data=normal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    anomal_metric = preprocess(data=anomal_metric, dataset=dataset, dk_select_useful=kwargs.get("dk_select_useful", False))
    intersect = [x for x in normal_metric.columns if x in anomal_metric.columns]
    normal_metric = normal_metric[intersect]
    anomal_metric = anomal_metric[intersect]

    # == logts ==
    logts = drop_constant(logts)
    normal_logts = logts[logts["time"] < inject_time].drop(columns=["time"])
    anomal_logts = logts[logts["time"] >= inject_time].drop(columns=["time"])

    # == traces_err ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        traces_err = traces_err.fillna(method='ffill')
        traces_err = traces_err.fillna(0)
        traces_err = drop_constant(traces_err)

        normal_traces_err = traces_err[traces_err["time"] < inject_time].drop(columns=["time"])
        anomal_traces_err = traces_err[traces_err["time"] >= inject_time].drop(columns=["time"])
    
     # == traces_lat ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        traces_lat = traces_lat.fillna(method='ffill')
        traces_lat = traces_lat.fillna(0)
        traces_lat = drop_constant(traces_lat)
        normal_traces_lat = traces_lat[traces_lat["time"] < inject_time].drop(columns=["time"])
        anomal_traces_lat = traces_lat[traces_lat["time"] >= inject_time].drop(columns=["time"])
    
    # ==== PROCESS ====
    ranks = []
    
    # == metric ==
    for col in normal_metric.columns:
        if col == "time":
            continue
        a = normal_metric[col].to_numpy()
        b = anomal_metric[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    # == logs ==
    for col in normal_logts.columns:
        a = normal_logts[col].to_numpy()
        b = anomal_logts[col].to_numpy()

        scaler = scaler_function().fit(a.reshape(-1, 1))
        zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
        score = max(zscores)
        ranks.append((col, score))

    # == traces_err ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        for col in normal_traces_err.columns:
            a = normal_traces_err[col].to_numpy()[:-2]
            b = anomal_traces_err[col].to_numpy()
                
            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))
   
    # == traces_lat ==
    if dataset == "mm-tt" or dataset == "mm-ob":
        for col in normal_traces_lat.columns:
            a = normal_traces_lat[col].to_numpy()
            b = anomal_traces_lat[col].to_numpy()

            scaler = scaler_function().fit(a.reshape(-1, 1))
            zscores = scaler.transform(b.reshape(-1, 1))[:, 0]
            score = max(zscores)
            ranks.append((col, score))

    ranks = sorted(ranks, key=lambda x: x[1], reverse=True)
    if kwargs.get("verbose") is True:
        for r, score in ranks[:20]:
            print(f"{r}: {score:.2f}")

    ranks = [x[0] for x in ranks]

    return {
        "ranks": ranks,
    }
    

def mmbaro(data, inject_time=None, dataset=None, num_loop=None, sli=None, anomalies=None, **kwargs):
    return mmnsigma(
        data=data,
        inject_time=inject_time,
        dataset=dataset,
        sli=sli,
        scaler_function=RobustScaler, **kwargs
    )
