# -*- coding: utf-8 -*-
"""三个 Task 共用的数据处理与评估工具。

负责分词、数据划分与指标计算，保证各 Task 使用完全一致的样本划分。
分词结果会缓存到 results/tokens_*.pkl，供后续脚本复用。
"""
import os
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# ---------------- 路径与全局常量 ----------------
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "HW-1")
RES_DIR = os.path.join(BASE, "results")
MODEL_DIR = os.path.join(BASE, "models")
NLTK_DIR = os.path.join(MODEL_DIR, "nltk_data")
os.makedirs(RES_DIR, exist_ok=True)

SEED = 42
DIM = 100            


def get_tokenizer():
    """返回 nltk.word_tokenize，优先使用项目内离线的 nltk_data。"""
    import nltk
    if os.path.isdir(NLTK_DIR):
        nltk.data.path.insert(0, NLTK_DIR)
    from nltk import word_tokenize
    return word_tokenize


def load_tokens(name):
    """载入分词结果，首次调用时执行分词并缓存。

    name 取 "nyt" 或 "ag"。nyt 返回 (tokens, labels)，ag 返回 (tokens, None)。
    word_tokenize 比按空格切分慢约 60 倍，全量约需 1 分钟，故缓存复用。
    """
    cache = os.path.join(RES_DIR, "tokens_%s.pkl" % name)
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            c = pickle.load(f)
        return c["tokens"], c.get("label")

    csv_path = os.path.join(DATA_DIR, "%s.csv" % name)
    cols = ["text", "label"] if name == "nyt" else ["text"]
    df = pd.read_csv(csv_path).dropna(subset=cols).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    word_tokenize = get_tokenizer()
    t0 = time.time()
    print("[%s] 分词中，共 %d 篇..." % (name, len(df)), flush=True)
    tokens = [word_tokenize(t) for t in df["text"]]
    print("[%s] 分词完成，用时 %.0f 秒" % (name, time.time() - t0))

    payload = {"tokens": tokens}
    if name == "nyt":
        payload["label"] = df["label"].tolist()
    with open(cache, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    return tokens, payload.get("label")


def get_split():
    """返回 NYT 的 8:1:1 划分结果。

    所有 Task 调用本函数获取划分，确保使用同一批样本。
    返回 (df, tr_idx, va_idx, te_idx, y_train, y_val, y_test, labels)。
    """
    tokens, label = load_tokens("nyt")
    df = pd.DataFrame({"tokens": tokens, "label": label})
    df["n_tok"] = df["tokens"].str.len()

    # 分层抽样，使三个子集的类别比例与全量一致
    train_df, temp_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"])
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

    return (df,
            train_df.index.values, val_df.index.values, test_df.index.values,
            train_df["label"].values, val_df["label"].values, test_df["label"].values,
            sorted(df["label"].unique()))


def evaluate(y_true, y_pred):
    """计算两项指标。"""
    return {"accuracy": accuracy_score(y_true, y_pred),
            "macro_f1": f1_score(y_true, y_pred, average="macro")}


def majority_baseline(y_train, y_test):
    """多数类基线：全部预测为训练集中样本最多的类别。"""
    maj = pd.Series(y_train).value_counts().idxmax()
    pred = np.full(len(y_test), maj)
    return evaluate(y_test, pred)


def print_split_info(df, tr_idx, va_idx, te_idx):
    print("数据划分（seed=%d）：" % SEED)
    for n, idx in [("train", tr_idx), ("val", va_idx), ("test", te_idx)]:
        sub = df.loc[idx, "label"].value_counts()
        print("  %-6s %5d 条 | %s" % (n, len(idx), dict(sub)))
