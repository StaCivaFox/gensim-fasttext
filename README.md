# Gensim 表示学习与 FastText 文本分类

项目仓库：[StaCivaFox/gensim-fasttext](https://github.com/StaCivaFox/gensim-fasttext)

使用 Gensim、scikit-learn 与 jieba 完成文本清洗、表示学习、文本分类、参数比较和模型保存/重载。代码包含英文新闻表示实验，以及以人民日报 9801 版面号为标签的 2/4/8 类分类流程。

## 环境与运行

依赖版本写在 `requirements.txt` 中。建议使用 Python 3.10–3.12 的独立虚拟环境：

```bash
git clone https://github.com/StaCivaFox/gensim-fasttext.git
cd gensim-fasttext
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell 使用 .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

将课程提供的人民日报 9801 原始语料保存为 `data/source_9801.txt.gz`，格式与转换方法见 [data/README.md](data/README.md)，再执行：

```bash
python run_all.py
```

该命令依次运行中文清洗、英文表示实验、中文分类和模型重载验证。处理后的文本、训练模型、实验指标分别生成在本地 `data/processed/`、`models/` 和 `results/`。重新训练会更新本地对应结果。

`fasttext-wheel` 是原生 fastText 的预编译分发，Python 导入名为 `fasttext`；依赖固定 NumPy 1.26.4，以匹配该分发版本。

## 单独运行与预测

英文实验使用 Gensim 安装包内的 Lee 新闻，可独立执行：

```bash
python -m experiment.representations
```

中文训练完成后，可以预测新文本并核验保存的模型：

```bash
python -m experiment.predict --method FastText_tuned --classes 8 --text "国际代表团就双边合作举行会谈。"
python -m experiment.predict --method Supervised_fastText --classes 4 --text "运动员在全国比赛中取得冠军。"
python -m experiment.verify
python -m experiment.source_audit
```

输出中的 `page_id` 表示预测版面号。源码审计可加 `--native-source /path/to/fasttext-0.9.2`，记录本地官方源码包的哈希。

## 实验配置

- 日期划分：1 月 1–21 日训练、22–26 日验证、27–31 日测试。
- 类别按 `[01,06]`、`[01,06,02,08]`、`[01,06,02,08,03,04,05,07]` 展开。
- 词表、IDF 和表示只从训练日期学习；分类器使用当前任务标签。
- 按验证集宏 F1 选择超参数，再评估测试集。
- 固定 `seed=42`、`PYTHONHASHSEED=42` 与单线程；LR 使用 `class_weight=balanced`。
- Gensim FastText 采用先学习词表示、再训练 LR 的路线；原生监督式 fastText 由标签联合优化嵌入与输出层。

## 代码结构

| 文件 | 作用 |
| --- | --- |
| `run_all.py` | 在独立子进程中依次运行实验阶段 |
| `experiment/prepare.py` | 文章聚合、文本清洗、分词和日期划分 |
| `experiment/features.py` | BoW、词向量均值、IDF 加权、Doc2Vec 推断 |
| `experiment/representations.py` | 英文八种表示、OOV 检查和定性检索 |
| `experiment/classify.py` | 中文分类、验证集选参和 FastText 参数比较 |
| `experiment/predict.py` | 从本地已训练模型预测新文本 |
| `experiment/verify.py` | 划分隔离、词表冻结与重载预测核验 |
| `experiment/source_audit.py` | 源码版本、哈希桶与本地错误统计 |
| `build_report.py` | 从本地结果文件生成 Markdown、PDF 和图表 |

## 生成实验报告

运行实验后，先补充源码与子词统计，再安装报告依赖并准备中文 TrueType 字体：

```bash
python -m experiment.source_audit
python -m pip install -r requirements-report.txt
python build_report.py --font /path/to/NotoSansSC-Regular.ttf
```

脚本读取本地 `results/` 与数据统计，输出 `REPORT.md`、PDF 和图表。Markdown 报告开头自动写入本仓库链接。报告中的叙述对应作业实验设计；更换语料或配置后，应结合新指标检查文字结论。

仓库发布内容为源码和运行说明。语料、模型、逐条预测、审计数据与实验报告在本地生成，并由 `.gitignore` 排除。

## 参考资料

- [Gensim 官网](https://radimrehurek.com/gensim/)
- [Enriching Word Vectors with Subword Information](https://aclanthology.org/Q17-1010/)
- [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/)
- [Gensim 4.4.0 源码](https://github.com/piskvorky/gensim/tree/4.4.0/gensim/models)
- [fastText v0.9.2 源码](https://github.com/facebookresearch/fastText/tree/v0.9.2/src)
