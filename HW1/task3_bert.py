# -*- coding: utf-8 -*-
"""Task 3：Fine-tune BERT。

按作业要求微调 google-bert/bert-base-uncased，max_length=64，训练 3 epoch，
在 NYT 测试集上报告 Accuracy 与 Macro-F1。

本机为 CPU 环境时单步训练约需 21 秒，3 epoch 约 10 小时，建议在 GPU 上运行。
脚本会自动检测设备，CPU 下会给出提示。

用法：
    python HW1/task3_bert.py                 # 完整训练
    python HW1/task3_bert.py --limit 200     # 小样本冒烟测试
"""
import argparse
import json
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

from common import RES_DIR, BASE, SEED, get_split, evaluate

warnings.filterwarnings("ignore")

# ---------------- 作业规定的设置 ----------------
MODEL_NAME = "google-bert/bert-base-uncased"
MAX_LENGTH = 64
EPOCHS = 3
# ---------------- 自选超参 ----------------
BATCH_SIZE = 32
LR = 2e-5
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# 若已在本地下载过模型则优先使用，否则从 Hugging Face 拉取
LOCAL_MODEL = os.path.join(BASE, "models", "bert", "bert-base-uncased")
MODEL_PATH = LOCAL_MODEL if os.path.isdir(LOCAL_MODEL) else MODEL_NAME


def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class NYTDataset(Dataset):
    """预先完成 tokenize，训练时直接取张量，避免每轮重复编码。"""

    def __init__(self, texts, labels, tokenizer, label2id):
        enc = tokenizer(list(texts), max_length=MAX_LENGTH, truncation=True,
                        padding="max_length", return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor([label2id[l] for l in labels], dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return {"input_ids": self.input_ids[i],
                "attention_mask": self.attention_mask[i],
                "labels": self.labels[i]}


@torch.no_grad()
def run_eval(model, loader, device):
    model.eval()
    preds, golds, losses = [], [], []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        out = model(**b)
        losses.append(out.loss.item())
        preds.append(out.logits.argmax(-1).cpu().numpy())
        golds.append(b["labels"].cpu().numpy())
    p, y = np.concatenate(preds), np.concatenate(golds)
    m = evaluate(y, p)
    m["loss"] = float(np.mean(losses))
    return m, p, y


def report_truncation(texts, tokenizer):
    """量化 max_length=64 造成的信息损失。"""
    sample = texts[:2000]
    lens = np.array([len(tokenizer.encode(t, add_special_tokens=True,
                                          truncation=False)) for t in sample])
    words = np.array([len(t.split()) for t in sample])
    print("\nmax_length=%d 的截断影响（抽样 %d 篇）：" % (MAX_LENGTH, len(sample)))
    print("  平均单词数        : %.0f" % words.mean())
    print("  平均 subword 数   : %.0f" % lens.mean())
    print("  subword/单词 比值 : %.2f" % (lens.mean() / words.mean()))
    print("  被截断的文本比例  : %.2f%%" % (100 * (lens > MAX_LENGTH).mean()))
    print("  平均保留内容比例  : %.2f%%"
          % (100 * np.minimum(lens, MAX_LENGTH).sum() / lens.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0,
                    help="仅使用前 N 条训练样本，用于快速验证流程")
    args = ap.parse_args()

    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("设备：%s" % device)
    if device.type == "cpu":
        print("提示：当前为 CPU 环境，完整训练约需 10 小时，建议改用 GPU，"
              "或加 --limit 参数先验证流程。")

    # 读取原始正文，BERT 使用自己的 tokenizer，不复用 nltk 分词结果
    raw = pd.read_csv(os.path.join(BASE, "HW-1", "nyt.csv")).dropna(
        subset=["text", "label"]).reset_index(drop=True)
    raw["text"] = raw["text"].astype(str).str.strip()
    raw = raw[raw["text"].str.len() > 0].reset_index(drop=True)

    # 复用 common 中的划分，保证与 Task 1、Task 2 使用同一批样本
    df, tr_idx, va_idx, te_idx, y_train, y_val, y_test, labels = get_split()
    assert len(raw) == len(df), "原始数据与分词缓存行数不一致"
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    if args.limit:
        tr_idx = tr_idx[:args.limit]
        va_idx, te_idx = va_idx[:64], te_idx[:64]
        y_train, y_val, y_test = y_train[:args.limit], y_val[:64], y_test[:64]
        print("冒烟模式：训练 %d 条" % len(tr_idx))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if not args.limit:
        report_truncation(raw["text"].tolist(), tokenizer)

    t0 = time.time()
    train_ds = NYTDataset(raw.loc[tr_idx, "text"], y_train, tokenizer, label2id)
    val_ds = NYTDataset(raw.loc[va_idx, "text"], y_val, tokenizer, label2id)
    test_ds = NYTDataset(raw.loc[te_idx, "text"], y_test, tokenizer, label2id)
    print("\n编码完成，用时 %.0f 秒" % (time.time() - t0))

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    set_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH, num_labels=len(labels),
        id2label=id2label, label2id=label2id).to(device)

    # bias 与 LayerNorm 不施加权重衰减，为 BERT 微调惯例
    no_decay = ["bias", "LayerNorm.weight"]
    grouped = [
        {"params": [p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)], "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(grouped, lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(WARMUP_RATIO * total_steps), total_steps)
    print("总步数 %d（warmup %d）\n" % (total_steps, int(WARMUP_RATIO * total_steps)))

    history = []
    t_start = time.time()
    for ep in range(1, EPOCHS + 1):
        model.train()
        run_loss, t_ep = 0.0, time.time()
        for i, b in enumerate(train_loader, 1):
            b = {k: v.to(device) for k, v in b.items()}
            loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            run_loss += loss.item()
            if i % 50 == 0 or i == len(train_loader):
                print("  epoch %d  step %3d/%d  loss %.4f"
                      % (ep, i, len(train_loader), run_loss / i), flush=True)
        vm, _, _ = run_eval(model, val_loader, device)
        history.append({"epoch": ep, "train_loss": run_loss / len(train_loader),
                        "val_loss": vm["loss"], "val_acc": vm["accuracy"],
                        "val_macro_f1": vm["macro_f1"], "sec": time.time() - t_ep})
        print("[epoch %d] train_loss=%.4f val_loss=%.4f val_Acc=%.4f val_MacroF1=%.4f (%.0fs)"
              % (ep, history[-1]["train_loss"], vm["loss"], vm["accuracy"],
                 vm["macro_f1"], history[-1]["sec"]), flush=True)

    print("\n训练完成，总用时 %.1f 分钟" % ((time.time() - t_start) / 60))

    tm, pred, gold = run_eval(model, test_loader, device)
    pred_lab = [id2label[i] for i in pred]
    gold_lab = [id2label[i] for i in gold]

    print("\n" + "=" * 62)
    print("Task 3 测试集结果 (n=%d)" % len(gold))
    print("=" * 62)
    print("  Accuracy : %.4f" % tm["accuracy"])
    print("  Macro-F1 : %.4f" % tm["macro_f1"])
    print("\n" + classification_report(gold_lab, pred_lab, digits=4, zero_division=0))
    cm = confusion_matrix(gold_lab, pred_lab, labels=labels)
    print("混淆矩阵（行=真实，列=预测）：")
    print(pd.DataFrame(cm, index=labels, columns=labels).to_string())

    pd.DataFrame(history).to_csv(os.path.join(RES_DIR, "bert_history.csv"), index=False)
    pd.DataFrame({"y_true": gold_lab, "pred_bert": pred_lab}).to_csv(
        os.path.join(RES_DIR, "test_predictions_bert.csv"),
        index=False, encoding="utf-8-sig")
    with open(os.path.join(RES_DIR, "bert_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"model": MODEL_NAME, "max_length": MAX_LENGTH, "epochs": EPOCHS,
                   "batch_size": BATCH_SIZE, "lr": LR, "seed": SEED,
                   "device": str(device),
                   "test_accuracy": tm["accuracy"], "test_macro_f1": tm["macro_f1"],
                   "confusion_matrix": cm.tolist(), "labels": labels,
                   "history": history}, f, ensure_ascii=False, indent=2)
    print("\n结果已保存至 results/bert_summary.json")


if __name__ == "__main__":
    main()
