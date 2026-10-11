# 代码说明

本目录包含三个 Task 的全部实现，均为可独立运行的 Python 脚本。

## 文件组成

| 文件 | 说明 |
|---|---|
| `common.py` | 共用模块：分词、数据划分、指标计算。三个 Task 均通过它获取划分结果，保证使用同一批样本 |
| `task1_bow.py` | Task 1，Binary Bag of Words 与 Word Frequency |
| `task2_embedding.py` | Task 2，GloVe 与两组自训 Word2Vec |
| `task3_bert.py` | Task 3，BERT 微调 |

## 环境依赖

```bash
pip install numpy pandas scikit-learn nltk gensim
pip install torch transformers        # 仅 Task 3 需要
```

开发环境为 Python 3.12.7，主要版本为 scikit-learn 1.5.1、gensim 4.3.3、
nltk 3.9.1、numpy 1.26.4、pandas 2.2.2。

## 数据与外部资源

仓库中需要存在以下内容，相对路径均以项目根目录为基准。

```
HW-1/nyt.csv          分类数据集
HW-1/ag.csv           训练 Word2Vec 的语料
models/glove/glove.6B.100d.txt     Task 2 需要
models/nltk_data/     可选，离线的 nltk 分词资源
```

GloVe 词向量需自行下载：

```bash
mkdir -p models/glove
curl -L -o models/glove/glove.6B.zip \
     https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
cd models/glove && unzip glove.6B.zip glove.6B.100d.txt
```

NLTK 分词器有两种获取方式。联网环境下直接执行：

```bash
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

若需离线使用，把 `punkt` 与 `punkt_tab` 解压到 `models/nltk_data/tokenizers/`，
`common.py` 会自动优先读取该目录。

## 运行方式

按顺序执行，Task 2 依赖 Task 1 生成的分词缓存。

```bash
python HW1/task1_bow.py          # 约 5 分钟
python HW1/task2_embedding.py    # 约 4 分钟
python HW1/task3_bert.py         # GPU 约 12 分钟，CPU 约 10 小时
```

从项目根目录或 `HW1/` 目录运行均可，脚本会自动定位项目路径。

Task 3 在 CPU 上耗时过长，实际在 GPU 环境运行。可用CPU先跑小样本的冒烟测试：

```bash
python HW1/task3_bert.py --limit 128
```

## 关于缓存

耗时较长的中间结果会写入项目根目录的 `results/`，再次运行时自动读取：

| 文件 | 内容 | 重算方式 |
|---|---|---|
| `tokens_nyt.pkl`、`tokens_ag.pkl` | 分词结果，全量约 1 分钟 | 删除文件 |
| `grid_search.csv` | Task 1 的超参搜索结果 | 删除文件 |
| `w2v_ag.kv`、`w2v_nyt.kv` | 自训词向量 | 删除文件 |

缓存词向量是因为 gensim 在 `workers>1` 时即使固定随机种子也不完全可复现，
实测两次运行的 Macro-F1 可相差 0.006。固定一次训练结果可使后续运行保持一致。

## 输出

运行结果统一写入 `results/`：

```
test_results.csv            Task 1 指标
test_predictions.csv        Task 1 逐条预测
task2_results.csv           Task 2 指标
docvec_*.npy                Task 2 文档向量
bert_summary.json           Task 3 指标与训练历史
bert_history.csv            Task 3 逐轮训练记录
test_predictions_bert.csv   Task 3 逐条预测
```

## 说明

- 随机种子统一为 42，数据按 8:1:1 分层划分，三个 Task 使用同一份划分。
