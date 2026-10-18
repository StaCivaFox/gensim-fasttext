# 实验数据

中文分类使用人民日报 1998 年 1 月语料，按原始文章编号中的版面号构建 2、4、8 类任务。

将课程提供的原始文本保存为 `data/source_9801.txt.gz` 后，在仓库根目录运行 `python run_all.py`。
如果拿到的是未压缩文件，可以使用 Python 标准库压缩：

```bash
python -c "import gzip,pathlib; gzip.open('data/source_9801.txt.gz','wb').write(pathlib.Path('你的9801文件.txt').read_bytes())"
```

清洗程序支持 UTF-8（含 BOM）与 GB18030，按如下记录结构解析：

```text
19980101-01-001-001 文章正文或一个段落
```

编号依次表示日期、版面、文章和段落。原始文本也可以单独处理：

```bash
python -m experiment.prepare --raw /path/to/9801.txt --output data/processed
```

运行后会在本地生成数据统计 `processed/stats.json`、清洗后的中文文本与模型。英文表示实验使用 Gensim 安装包内的 Lee 新闻文件，无需单独下载。
