# -*- coding: utf-8 -*-
"""Task 2：Word Embedding。

三组实验统一使用 100 维词向量，文档向量取有效词向量的平均，分类器为
Logistic Regression：
  2.1 预训练 GloVe (glove.6B.100d)
  2.2 在 AG News 上自训 Word2Vec
  2.3 在 NYT 上自训 Word2Vec

gensim 在 workers>1 时即使固定 seed 也不完全可复现，故训练后的词向量会
保存到 results/w2v_*.kv，重跑时直接加载。删除该文件可强制重新训练。

用法：python HW1/task2_embedding.py
"""
import os
import time
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from common import (RES_DIR, MODEL_DIR, SEED, DIM,
                    get_split, load_tokens, evaluate, majority_baseline)

GLOVE_TXT = os.path.join(MODEL_DIR, "glove", "glove.6B.100d.txt")
C_GRID = [0.01, 0.1, 1.0, 10.0, 100.0]
W2V_KW = dict(vector_size=DIM, window=5, min_count=5, sg=0,   # sg=0 为 CBOW
              workers=4, epochs=5, seed=SEED)


def build_doc_vectors(tokens_list, lookup, dim=DIM):
    """文档向量 = 该文档中所有能查到词向量的词的平均，查不到的跳过。"""
    D = np.zeros((len(tokens_list), dim), dtype=np.float32)
    n_tok = n_hit = 0
    oov = set()
    for i, toks in enumerate(tokens_list):
        vecs = []
        for w in toks:
            n_tok += 1
            if w in lookup:
                vecs.append(lookup[w]); n_hit += 1
            else:
                oov.add(w)
        if vecs:
            D[i] = np.mean(vecs, axis=0)
    return D, {"token_cov": n_hit / max(n_tok, 1), "n_oov_types": len(oov)}


def load_glove(vocab):
    """只载入 NYT 词表内的词。完整载入 40 万词约需 2.4 GB 内存，
    未出现在 NYT 中的词本实验不会查询，筛除不影响结果。"""
    if not os.path.exists(GLOVE_TXT):
        raise FileNotFoundError(
            "缺少 %s\n请先下载：\n"
            "  curl -L -o models/glove/glove.6B.zip "
            "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip\n"
            "  cd models/glove && unzip glove.6B.zip glove.6B.100d.txt" % GLOVE_TXT)
    t0 = time.time()
    g = {}
    with open(GLOVE_TXT, encoding="utf-8") as f:
        for line in f:
            sp = line.find(" ")
            w = line[:sp]
            if w in vocab:
                g[w] = np.fromstring(line[sp + 1:], dtype=np.float32, sep=" ")
    print("  载入 GloVe %d 词（NYT 词表的 %.1f%%），用时 %.0f 秒"
          % (len(g), 100 * len(g) / len(vocab), time.time() - t0))
    return g


def train_w2v(name, sentences):
    """训练或载入 Word2Vec 词向量。"""
    path = os.path.join(RES_DIR, "w2v_%s.kv" % name)
    if os.path.exists(path):
        wv = KeyedVectors.load(path)
        print("  复用已训练模型 %s，词表 %d" % (path, len(wv)))
        return wv
    t0 = time.time()
    model = Word2Vec(sentences=sentences, **W2V_KW)
    model.wv.save(path)
    print("  训练完成，词表 %d，用时 %.0f 秒" % (len(model.wv), time.time() - t0))
    return model.wv


def run_experiment(name, D, tr_idx, va_idx, te_idx, y_train, y_val, y_test):
    """在验证集上选 C，再在测试集上评估一次。"""
    best_C, best_f1 = None, -1.0
    for C in C_GRID:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=C, max_iter=3000,
                                     random_state=SEED).fit(D[tr_idx], y_train)
        f1 = evaluate(y_val, clf.predict(D[va_idx]))["macro_f1"]
        if f1 > best_f1:
            best_f1, best_C = f1, C
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(C=best_C, max_iter=3000,
                                 random_state=SEED).fit(D[tr_idx], y_train)
    pred = clf.predict(D[te_idx])
    m = evaluate(y_test, pred)
    print("  -> 最优 C=%-5g 验证 Macro-F1=%.4f | 测试 Accuracy=%.4f Macro-F1=%.4f"
          % (best_C, best_f1, m["accuracy"], m["macro_f1"]))
    return dict(experiment=name, best_C=best_C, **m), pred


def main():
    df, tr_idx, va_idx, te_idx, y_train, y_val, y_test, labels = get_split()
    nyt_tokens = df["tokens"].tolist()
    nyt_vocab = set()
    for t in nyt_tokens:
        nyt_vocab.update(t)
    print("NYT 共 %d 篇，词表 %d 个不同词\n" % (len(df), len(nyt_vocab)))

    results, preds, stats, vocabs = [], {}, {}, {}

    # ---------------- 2.1 预训练 GloVe ----------------
    print("[2.1] Pre-trained GloVe")
    glove = load_glove(nyt_vocab)
    D, st = build_doc_vectors(nyt_tokens, glove)
    stats["GloVe"], vocabs["GloVe"] = st, len(glove)
    print("  token 覆盖率 %.2f%%" % (100 * st["token_cov"]))
    r, p = run_experiment("GloVe", D, tr_idx, va_idx, te_idx, y_train, y_val, y_test)
    results.append(r); preds["GloVe"] = p
    np.save(os.path.join(RES_DIR, "docvec_GloVe.npy"), D)
    del D

    # ---------------- 2.2 Word2Vec on AG News ----------------
    print("\n[2.2] Word2Vec on AG News")
    ag_tokens, _ = load_tokens("ag")
    print("  AG News 共 %d 篇，合计 %d token"
          % (len(ag_tokens), sum(len(t) for t in ag_tokens)))
    wv_ag = train_w2v("ag", ag_tokens)
    del ag_tokens
    D, st = build_doc_vectors(nyt_tokens, wv_ag)
    stats["W2V-AG"], vocabs["W2V-AG"] = st, len(wv_ag)
    print("  token 覆盖率 %.2f%%" % (100 * st["token_cov"]))
    r, p = run_experiment("W2V-AG", D, tr_idx, va_idx, te_idx, y_train, y_val, y_test)
    results.append(r); preds["W2V-AG"] = p
    np.save(os.path.join(RES_DIR, "docvec_W2V_AG.npy"), D)
    del D

    # ---------------- 2.3 Word2Vec on NYT ----------------
    print("\n[2.3] Word2Vec on NYT")
    print("  NYT 合计 %d token" % sum(len(t) for t in nyt_tokens))
    wv_nyt = train_w2v("nyt", nyt_tokens)
    D, st = build_doc_vectors(nyt_tokens, wv_nyt)
    stats["W2V-NYT"], vocabs["W2V-NYT"] = st, len(wv_nyt)
    print("  token 覆盖率 %.2f%%" % (100 * st["token_cov"]))
    r, p = run_experiment("W2V-NYT", D, tr_idx, va_idx, te_idx, y_train, y_val, y_test)
    results.append(r); preds["W2V-NYT"] = p
    np.save(os.path.join(RES_DIR, "docvec_W2V_NYT.npy"), D)
    del D

    # ---------------- 汇总 ----------------
    base = majority_baseline(y_train, y_test)
    res_df = pd.DataFrame(
        [dict(experiment="majority-baseline", best_C=np.nan, **base)] + results)
    res_df.to_csv(os.path.join(RES_DIR, "task2_results.csv"), index=False)

    print("\n" + "=" * 70)
    print("Task 2 测试集结果 (n=%d)" % len(y_test))
    print("=" * 70)
    print(res_df.round(4).to_string(index=False))

    print("\n覆盖率与词表规模：")
    cov = pd.DataFrame([{"实验": k, "词向量词表": vocabs[k],
                         "token覆盖率%": round(100 * v["token_cov"], 2),
                         "未覆盖词种类": v["n_oov_types"]}
                        for k, v in stats.items()])
    print(cov.to_string(index=False))

    for name in ["GloVe", "W2V-AG", "W2V-NYT"]:
        print("\n【%s】分类报告" % name)
        print(classification_report(y_test, preds[name], digits=4, zero_division=0))

    # 词向量质量抽查
    print("最近邻抽查：")
    for w in ["bank", "court", "football"]:
        for nm, wv in [("W2V-AG", wv_ag), ("W2V-NYT", wv_nyt)]:
            if w in wv:
                sim = ", ".join(x for x, _ in wv.most_similar(w, topn=5))
                print("  %-9s %-8s %s" % (w, nm, sim))
    print("\n结果已保存至 results/task2_results.csv")


if __name__ == "__main__":
    main()
