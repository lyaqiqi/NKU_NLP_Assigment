# -*- coding: utf-8 -*-
"""Task 1：Bag of Words。

实现 Binary Bag of Words 与 Word Frequency 两种特征，分类器均为 Logistic
Regression，在 NYT 测试集上报告 Accuracy 与 Macro-F1。

两种特征共用同一词表：先计算词频矩阵，再在同一矩阵上二值化得到 Binary 版本，
以保证词表与列序完全一致，差异仅在取值方式。

用法：python HW1/task1_bow.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from common import RES_DIR, SEED, get_split, evaluate, majority_baseline, print_split_info

C_GRID = [0.01, 0.1, 1.0, 10.0]
MAX_ITER = 1000


def build_features(df, tr_idx):
    """构建两种词袋特征。词表仅在训练集上拟合。"""
    t0 = time.time()
    # 文本已预分词，analyzer 直接返回 token 列表
    vec = CountVectorizer(analyzer=lambda toks: toks, min_df=1)
    vec.fit(df.loc[tr_idx, "tokens"])

    X_count = vec.transform(df["tokens"]).astype(np.float32)
    X_bin = (X_count > 0).astype(np.float32)   # 同一矩阵二值化

    print("特征构建完成，用时 %.1f 秒" % (time.time() - t0))
    print("  词表大小     : %d" % len(vec.vocabulary_))
    print("  训练矩阵形状 : (%d, %d)" % (len(tr_idx), X_count.shape[1]))
    print("  稀疏度       : %.4f%%"
          % (100.0 * X_count[tr_idx].nnz / (len(tr_idx) * X_count.shape[1])))
    # 统计口径为训练集，与词表的构建范围保持一致
    print("  取值范围     : Binary [%.0f, %.0f] | 词频 [%.0f, %.0f]"
          % (X_bin[tr_idx].min(), X_bin[tr_idx].max(),
             X_count[tr_idx].min(), X_count[tr_idx].max()))
    print("  单篇各维之和 : Binary %.1f | 词频 %.1f"
          % (X_bin[tr_idx].sum(axis=1).mean(), X_count[tr_idx].sum(axis=1).mean()))
    return vec, X_bin, X_count


def search_C(X, tr_idx, va_idx, y_train, y_val, tag):
    """在验证集上搜索正则强度，同时记录迭代次数。"""
    rows = []
    for C in C_GRID:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=C, max_iter=MAX_ITER,
                                     random_state=SEED).fit(X[tr_idx], y_train)
        m_val = evaluate(y_val, clf.predict(X[va_idx]))
        m_tr = evaluate(y_train, clf.predict(X[tr_idx]))
        rows.append({"feature": tag, "C": C,
                     "train_acc": m_tr["accuracy"], "train_f1": m_tr["macro_f1"],
                     "val_acc": m_val["accuracy"], "val_f1": m_val["macro_f1"],
                     "n_iter": int(np.max(clf.n_iter_)), "fit_sec": time.time() - t0})
        print("  [%-6s] C=%-6g val_Acc=%.4f val_MacroF1=%.4f 迭代=%-4d (%.1fs)"
              % (tag, C, m_val["accuracy"], m_val["macro_f1"],
                 rows[-1]["n_iter"], rows[-1]["fit_sec"]), flush=True)
    return rows


def main():
    df, tr_idx, va_idx, te_idx, y_train, y_val, y_test, labels = get_split()
    print_split_info(df, tr_idx, va_idx, te_idx)
    print()

    vec, X_bin, X_count = build_features(df, tr_idx)
    feats = {"binary": X_bin, "count": X_count}

    # ---------------- 超参搜索 ----------------
    cache = os.path.join(RES_DIR, "grid_search.csv")
    if os.path.exists(cache):
        grid = pd.read_csv(cache)
        print("\n复用已有搜索结果：%s（删除该文件可强制重算）" % cache)
    else:
        print("\n超参搜索：")
        rows = []
        for tag, X in feats.items():
            rows += search_C(X, tr_idx, va_idx, y_train, y_val, tag)
        grid = pd.DataFrame(rows)
        grid.to_csv(cache, index=False)
    best = {t: grid[grid.feature == t].sort_values("val_f1").iloc[-1]
            for t in feats}
    print("\n最优配置：")
    for t, r in best.items():
        print("  %-7s C=%-6g val_MacroF1=%.4f 迭代=%d"
              % (t, r.C, r.val_f1, int(r.n_iter)))

    # ---------------- 测试集评估 ----------------
    results = [dict(feature="majority-baseline", C=np.nan,
                    **majority_baseline(y_train, y_test))]
    preds = {}
    print("\n测试集评估：")
    for tag, X in feats.items():
        C = float(best[tag].C)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=C, max_iter=MAX_ITER,
                                     random_state=SEED).fit(X[tr_idx], y_train)
        pred = clf.predict(X[te_idx])
        preds[tag] = pred
        m = evaluate(y_test, pred)
        results.append(dict(feature=tag, C=C, **m))
        print("  %-7s C=%-6g Accuracy=%.4f Macro-F1=%.4f"
              % (tag, C, m["accuracy"], m["macro_f1"]))

    res_df = pd.DataFrame(results)
    res_df.to_csv(os.path.join(RES_DIR, "test_results.csv"), index=False)

    print("\n" + "=" * 62)
    print("Task 1 测试集结果 (n=%d)" % len(y_test))
    print("=" * 62)
    print(res_df.round(4).to_string(index=False))

    for tag in feats:
        print("\n【%s】分类报告" % tag)
        print(classification_report(y_test, preds[tag], digits=4, zero_division=0))

    pd.DataFrame({"y_true": y_test,
                  "pred_binary": preds["binary"],
                  "pred_count": preds["count"],
                  "n_tok": df.loc[te_idx, "n_tok"].values}
                 ).to_csv(os.path.join(RES_DIR, "test_predictions.csv"),
                          index=False, encoding="utf-8-sig")
    print("结果已保存至 results/test_results.csv 与 results/test_predictions.csv")


if __name__ == "__main__":
    main()
