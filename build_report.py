"""Build a focused Chinese report from stored experimental evidence.

Usage: python build_report.py --font /path/to/NotoSansSC-Regular.ttf
Requires matplotlib, numpy and reportlab; does not retrain any model.
"""
import argparse
from html import escape
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

ROOT = Path(__file__).resolve().parent
REPOSITORY_URL = 'https://github.com/StaCivaFox/gensim-fasttext'

def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))

def pct(value):
    return f'{100 * value:.2f}%'

def build(font):
    result = read('results/classification.json')
    lookup = {(r['classes'], r['method']): r for r in result}
    stats = read('data/processed/stats.json')
    lee = read('results/lee_representations.json')
    ablation = read('results/fasttext_ablation.json')
    audit = read('results/source_audit.json')
    config = read('results/run_config.json')
    verify = read('results/verification.json')
    coverage = read('results/coverage.json')
    base = lookup[8, 'FastText_tfidf']
    tuned = lookup[8, 'FastText_tuned']
    gain = 100 * (tuned['macro_f1'] - base['macro_f1'])
    chosen = max(ablation, key=lambda a: a['dev_macro_f1'])
    english = lee['coverage']['FastText']['test']
    chinese = coverage['FastText_tuned']['test']
    en_w2v = lee['coverage']['Word2Vec_SG']['test']
    zh_w2v = coverage['Word2Vec_mean']['test']
    lengths = audit['vocabulary_word_lengths']
    short_words = sum(v for k, v in lengths.items() if int(k) <= 3)
    short_fraction = short_words / sum(lengths.values())
    names = {
        'Majority': '多数类基线', 'BoW': 'BoW + LR', 'TFIDF': 'TF-IDF + LR',
        'LSI': 'LSI + LR', 'Word2Vec_mean': 'Word2Vec 均值 + LR',
        'FastText_mean': 'FastText 均值 + LR（3–6）',
        'FastText_tfidf': 'FastText 加权 + LR（3–6）',
        'FastText_tuned': 'FastText 加权 + LR（1–3）',
        'Doc2Vec': 'Doc2Vec + LR', 'Supervised_fastText': '监督式 fastText',
    }
    figures = ROOT / 'results/figures'
    figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(7.6, 2.6))
    x = np.arange(2)
    w2v = [100 * en_w2v['represented_tokens'] / en_w2v['tokens'],
           100 * zh_w2v['represented_tokens'] / zh_w2v['tokens']]
    ft = [100 * english['represented_tokens'] / english['tokens'],
          100 * chinese['represented_tokens'] / chinese['tokens']]
    for offset, values, label, color in [(-.18, w2v, 'Word2Vec', '#7b8995'),
                                         (.18, ft, 'FastText', '#147f86')]:
        bars = ax.bar(x + offset, values, width=.31, label=label, color=color)
        ax.bar_label(bars, labels=[f'{v:.2f}%' for v in values], padding=3, fontsize=10)
    ax.set(xticks=x, xticklabels=[f'English news ({english["tokens"]:,} tokens)', f'Chinese news ({chinese["tokens"]:,} tokens)'],
           ylabel='Token vector coverage (%)', ylim=(0, 113), yticks=[0, 25, 50, 75, 100])
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, 1.22), ncol=2, frameon=False)
    fig.tight_layout(pad=.8)
    fig.savefig(figures / 'token_coverage.png', dpi=180)
    plt.close(fig)

    # Keep the full classification figure available in the reproducibility package.
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    for method, label, color in [('TFIDF', 'TF-IDF + LR', '#187c85'),
                                ('FastText_tfidf', 'FastText weighted 3-6 + LR', '#ca7a2c'),
                                ('FastText_tuned', 'FastText weighted 1-3 + LR', '#6d55a3'),
                                ('Supervised_fastText', 'Supervised fastText', '#31485b')]:
        ax.plot([2, 4, 8], [100 * lookup[k, method]['macro_f1'] for k in [2, 4, 8]],
                'o-', color=color, label=label, lw=2, ms=5)
    ax.set(xticks=[2, 4, 8], xlabel='Number of page labels', ylabel='Test macro F1 (%)', ylim=(0, 100))
    ax.grid(axis='y', alpha=.2)
    ax.legend(ncol=2, fontsize=8, loc='lower left', frameon=False)
    fig.tight_layout()
    fig.savefig(figures / 'classification_macro_f1.png', dpi=180)
    plt.close(fig)

    cm = np.array(tuned['confusion_matrix'])
    norm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    im = ax.imshow(norm, cmap='Blues', vmin=0, vmax=1)
    ax.set(xticks=range(8), yticks=range(8), xticklabels=tuned['labels'], yticklabels=tuned['labels'],
           xlabel='Predicted page', ylabel='True page')
    for i in range(8):
        for j in range(8):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=9,
                    color='white' if norm[i, j] > .5 else '#253a4e')
    fig.colorbar(im, ax=ax, label='Row-normalized fraction', pad=.04)
    fig.tight_layout()
    fig.savefig(figures / 'fasttext_confusion_8.png', dpi=180)
    plt.close(fig)

    pdfmetrics.registerFont(TTFont('CJK', str(font)))
    pdfmetrics.registerFontFamily('CJK', normal='CJK', bold='CJK', italic='CJK', boldItalic='CJK')
    blue = colors.HexColor('#17364d')
    teal = colors.HexColor('#147f86')
    style = {
        'title': ParagraphStyle('title', fontName='CJK', fontSize=24, leading=33, textColor=blue, spaceAfter=11, wordWrap='CJK'),
        'subtitle': ParagraphStyle('subtitle', fontName='CJK', fontSize=12, leading=19, textColor=teal, spaceAfter=13, wordWrap='CJK'),
        'h1': ParagraphStyle('h1', fontName='CJK', fontSize=17, leading=25, textColor=blue, spaceAfter=12, wordWrap='CJK'),
        'h2': ParagraphStyle('h2', fontName='CJK', fontSize=11.5, leading=18, textColor=teal, spaceBefore=8, spaceAfter=6, wordWrap='CJK'),
        'body': ParagraphStyle('body', fontName='CJK', fontSize=9.4, leading=15.7, spaceAfter=8, wordWrap='CJK'),
        'small': ParagraphStyle('small', fontName='CJK', fontSize=8.2, leading=12.3, spaceAfter=6, wordWrap='CJK'),
        'cell': ParagraphStyle('cell', fontName='CJK', fontSize=8.2, leading=12.1, wordWrap='CJK'),
        'head': ParagraphStyle('head', fontName='CJK', fontSize=8.2, leading=12.1, textColor=colors.white, wordWrap='CJK'),
    }
    title = '子词表示的覆盖与中文适配'
    story, md = [], [f'项目仓库：[StaCivaFox/gensim-fasttext]({REPOSITORY_URL})\n', f'# {title}\n', 'Gensim 表示学习与 FastText 文本分类实验报告\n']
    def p(text, kind='body'):
        story.append(Paragraph(escape(text).replace('\n', '<br/>'), style[kind]))
        md.append(text + '\n')
    def h(text):
        story.append(Paragraph(escape(text), style['h1']))
        md.append('\n## ' + text + '\n')
    def sub(text):
        story.append(Paragraph(escape(text), style['h2']))
        md.append('\n### ' + text + '\n')
    def table(headers, rows, widths):
        data = [[Paragraph(escape(str(v)), style['head']) for v in headers]]
        data += [[Paragraph(escape(str(v)).replace('\n', '<br/>'), style['cell']) for v in row] for row in rows]
        t = Table(data, colWidths=widths, repeatRows=1, hAlign='LEFT')
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), blue), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('LEFTPADDING', (0, 0), (-1, -1), 7), ('RIGHTPADDING', (0, 0), (-1, -1), 7),
            ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f5f7'), colors.white]),
            ('LINEBELOW', (0, -1), (-1, -1), .4, colors.HexColor('#ccd9df')),
        ]))
        story.extend([t, Spacer(1, 7)])
        md.extend(['| ' + ' | '.join(map(str, headers)) + ' |', '|' + '---|' * len(headers)])
        md.extend('| ' + ' | '.join(str(v).replace('|', '/').replace('\n', '<br>') for v in row) + ' |' for row in rows)
        md.append('')
    def picture(filename, width, height, caption):
        story.append(Image(str(figures / filename), width=width, height=height))
        story.append(Paragraph(escape(caption), style['small']))
        md.append(f'![{caption}](results/figures/{filename})\n')
    def page():
        story.append(PageBreak())

    # Main text, page 1: problem, approach, and the strongest supported findings.
    p(title, 'title')
    p('Gensim 表示学习与 FastText 文本分类实验报告', 'subtitle')
    p('可选编程作业  /  中英文新闻语料  /  2026-10-18', 'small')
    sub('摘要')
    p('新闻文本中的新词和低频词要求表示模型能够处理训练词表之外的输入；中文短词结构又使字符片段长度成为值得实测的参数。本文以 Gensim 构建从文本清洗、表示学习到分类评估的完整流程，在英文新闻与人民日报 9801 语料上考察子词覆盖能力，并通过受控参数比较选择适合中文任务的字符粒度。')
    p(f'实验中，FastText 在英文与中文测试集上的词次向量覆盖率均为 100%，分别覆盖 {english["oov_tokens"]:,} 和 {chinese["oov_tokens"]:,} 个词表外词次。在同为 50 维、20,000 个哈希桶的设置下，验证集选出的 1–3 字符模型使中文八类测试宏 F1 从 {pct(base["macro_f1"])} 提升至 {pct(tuned["macro_f1"])}，增加 {gain:.2f} 个百分点。结果将子词共享的输入覆盖能力与字符粒度的下游作用联系起来，为中文新闻的表示选型提供了可复现的依据。')
    sub('核心结果')
    table(['观察维度', '实测证据', '对应价值'], [
        ['测试词次覆盖', f'英文 {w2v[0]:.2f}% → {ft[0]:.2f}%\n中文 {w2v[1]:.2f}% → {ft[1]:.2f}%', '以共享字符片段补充整词查表的输入覆盖。'],
        ['中文字符粒度', f'八类宏 F1 +{gain:.2f} 个百分点\n子词矩阵均为 3.81 MiB', '在固定子词矩阵容量下，通过粒度选择改善分类。'],
        ['流程与实现', '8 种英文表示；2/4/8 类分类\n5 组参数；两条 fastText 路线', '将模型行为、论文机制和实现位置对应起来。'],
    ], [91, 184, 236])
    sub('研究问题与实验组织')
    p('本文依次回答三个问题：子词表示能为多少测试词次提供向量；中文字符范围的改变能否转化为分类增益；无监督词表示与标签驱动的嵌入学习如何对应两篇 FastText 论文。正文围绕这些问题组织证据，完整方法对比、类别诊断和复现配置见附录。')
    p(f'英文采用 Gensim 随包提供的 Lee 新闻，训练 300 篇、测试 50 篇。中文采用人民日报 9801，清洗后共 {sum(stats[s]["articles"] for s in ["train", "dev", "test"]):,} 篇，以原始版面号作为类别，按日期划分训练、验证和测试。两种语言独立建模；全部模型采用 seed=42 的单次训练，参数由验证集宏 F1 选择。', 'small')

    # Main text, page 2: availability evidence and representation-learning breadth.
    page(); h('01  子词共享扩大测试输入的覆盖')
    p('词表外词（OOV）在新闻中可能来自新实体、低频表达或词形变化。Word2Vec 通过已学习的整词索引查找向量；FastText 进一步共享字符 n-gram，使词表外输入也能通过子词组合获得表示 [1,3]。这里以“获得有限、非零词向量的测试词次占比”衡量覆盖率，统计单位为词的出现次数。')
    picture('token_coverage.png', 490, 168,
            f'图 1  测试词次向量覆盖率。英文 {english["tokens"]:,} 词次，中文八类 {chinese["tokens"]:,} 词次；覆盖率衡量向量可计算性。')
    table(['测试语料', '词表外词次', 'Word2Vec 可表示', 'FastText 可表示'], [
        ['英文 Lee 新闻', f'{english["oov_tokens"]:,}', f'{en_w2v["represented_tokens"]:,}', f'{english["represented_tokens"]:,}'],
        ['中文八类新闻', f'{chinese["oov_tokens"]:,}', f'{zh_w2v["represented_tokens"]:,}', f'{chinese["represented_tokens"]:,}'],
    ], [139, 111, 130, 131])
    p('两份测试集上，FastText 都为整词词表之外的词次保留了可用的数值输入。覆盖率回答输入能否参与表示计算；其对任务判别的贡献由后续分类指标评价。中文文档以词向量均值或正 IDF 加权均值构造，再做 L2 归一化，OOV 在加权分支中使用训练集的最大 IDF。')
    sub('在统一新闻语料上理解不同表示')
    table(['英文表示', '训练矩阵', '非零占比', '训练及转换 / 秒'], [
        [r['method'], ' × '.join(map(str, r['train_shape'])), pct(r['nonzero_fraction']), f'{r["fit_and_transform_seconds"]:.2f}']
        for r in lee['summary']
    ], [165, 112, 94, 140])
    p('BoW 与 TF-IDF 保留显式词特征；LSI 将 TF-IDF 投影到 50 维；LDA 产生 10 维主题分布；Word2Vec、FastText 和 Doc2Vec 提供 50 维密集表示。检索实验对政治、山火和体育三组查询保存各方法的 top-3 文档及余弦分数，用于定性观察表示差异。计时反映本次运行环境，完整主题词与检索结果见 lee_representations.json。', 'small')

    # Main text, page 3: same-budget controlled comparison and interpretable source diagnostics.
    page(); h('02  以中文词结构选择字符粒度')
    p(f'中文训练词表共有 {sum(lengths.values()):,} 个词型，其中 1–3 字词占 {pct(short_fraction)}。这一结构为比较较短字符片段提供了依据。本文固定训练文本、随机种子和训练轮数，以 3–6 字符的 Skip-gram 模型为起点，分别改变字符长度、哈希桶数、维数或训练结构。')
    table(['配置', '维数', '字符范围', '桶数', '结构', '验证宏 F1'], [
        [a['name'], a['dim'], f'{a["min_n"]}–{a["max_n"]}', f'{a["bucket"]:,}', 'SG' if a['sg'] else 'CBOW', pct(a['dev_macro_f1'])]
        for a in ablation
    ], [122, 44, 72, 74, 61, 138])
    p('各组均训练 10 轮，以同一 IDF 加权和 L2 归一化方式构造文档表示；LR 的 C 在 {0.1, 1, 10} 中独立选择。表中五组最终均选择 C=10，字符范围对照因而保持了相同的分类器正则参数。')
    sub('固定容量下，字符粒度的选择转化为测试增益')
    table(['字符范围', '验证宏 F1', '八类测试宏 F1', '子词矩阵'], [
        ['3–6（基础）', pct(ablation[0]['dev_macro_f1']), pct(base['macro_f1']), '3.81 MiB'],
        ['1–3（验证集选中）', pct(chosen['dev_macro_f1']), pct(tuned['macro_f1']), '3.81 MiB'],
    ], [151, 110, 130, 120])
    p(f'验证集选择 1–3 字符模型后，独立测试集宏 F1 提高 {gain:.2f} 个百分点。两组保持 50 维和 20,000 桶，子词矩阵的分配规模相同，因此这项增益来自字符粒度配置的改变，而没有增加子词矩阵容量。对于这份中文新闻语料，字符范围是一个有效的调参方向。')
    sub('哈希统计将参数选择连接到实现机制')
    table(['字符范围 / 桶数', '不同子词串', '占用桶', '碰撞比例', '矩阵 MiB'], [
        [f'{a["min_n"]}–{a["max_n"]} / {a["bucket"]:,}', f'{a["unique_character_ngrams"]:,}', f'{a["occupied_buckets"]:,}', pct(a['collision_fraction']), f'{a["ngram_matrix_mib_float32_dim50"]:.2f}']
        for a in audit['hash_collisions']
    ], [147, 100, 83, 95, 86])
    p('1–3 字符配置产生较少的不同子词串，20,000 桶上的碰撞比例也随之降低。这与中文短词占比较高的词表结构相呼应，为结果提供了实现层面的解释；当前对照识别的是字符范围配置的整体作用，碰撞和特征内容各自的贡献仍需独立控制实验分解。')
    p('碰撞比例定义为 1 − 占用桶数 / 不同子词字符串数，统计范围为保留的训练词表。矩阵容量按 bucket × dim × 4 字节计算，指 float32 子词矩阵。“中国”的 3–6 字符片段包括“<中国”“<中国>”“中国>”，边界标记也参与字符片段构造 [5]。', 'small')

    # Main text, page 4: task context, paper distinction, and a forward conclusion.
    page(); h('03  从表示学习到标签驱动的分类')
    p('为把表示实验落实到下游任务，本文依次建立 2、4、8 个版面标签的分类任务，并同时训练原生监督式 fastText。前一条路线先学习上下文，再用冻结向量训练逻辑回归；后一条路线直接由标签更新输入嵌入和分类层。两条路线对应不同的训练目标 [3,4]。')
    table(['实验路线', '输入与训练目标', '标签对表示的作用'], [
        ['Gensim FastText + LR', '字符子词与整词预测上下文；文档采用加权均值。', '分类阶段冻结词向量，标签训练 LR。'],
        ['原生监督式 fastText', '词或词 n-gram 嵌入平均，优化标签交叉熵。', '标签同时更新嵌入与输出矩阵。'],
    ], [145, 207, 159])
    sub('分类结果提供独立的任务评价')
    table(['方法', '2 类宏 F1', '4 类宏 F1', '8 类宏 F1'], [
        [names[m]] + [pct(lookup[k, m]['macro_f1']) for k in [2, 4, 8]]
        for m in ['FastText_tfidf', 'FastText_tuned', 'Supervised_fastText', 'TFIDF']
    ], [232, 93, 93, 93])
    p(f'监督式 fastText 在三项任务上分别达到 {pct(lookup[2,"Supervised_fastText"]["macro_f1"])}、{pct(lookup[4,"Supervised_fastText"]["macro_f1"])} 和 {pct(lookup[8,"Supervised_fastText"]["macro_f1"])}。TF-IDF + LR 提供稀疏词特征参照，其八类宏 F1 为 {pct(lookup[8,"TFIDF"]["macro_f1"])}。这些结果共同刻画了本任务中不同学习路线的表现；本文关于中文适配的直接证据来自同一 FastText 路线内的字符范围对照。')
    sub('论文中的运算如何落到源码')
    p('子词论文将一个词分解为共享字符片段，再用组合表示预测上下文 [3]。Gensim 的 ft_ngram_hashes 完成字符片段的哈希取桶，fasttext_fast_sentence_sg_neg 更新整词、子词和上下文参数，get_vector 在推断时组合向量 [5]。这条调用链解释了词表外输入为何仍能获得表示。')
    p('分类论文以特征嵌入的平均值连接线性输出层，简写为 h = A x / |x|，p(y|x) = softmax(B h) [4]。原生源码的 computeHidden 与 update 分别完成平均表示和联合更新，addWordNgrams 提供跨词特征 [6]。本实验各类别任务均由验证集选中 lr=0.3、wordNgrams=1，固定 dim=50、epoch=25、loss=softmax。')
    sub('结论')
    p(f'本实验将 FastText 的子词机制转化为两项可检查的结果：中英文新闻测试词次均可获得向量表示；在固定子词矩阵容量下，中文字符粒度选择带来 {gain:.2f} 个百分点的八类测试宏 F1 增益。由此形成一条明确的实践路径：先检查词表外输入的覆盖，再依据训练词结构设置候选字符范围，以验证集选择配置，最后用独立测试集评价分类作用。配套的训练代码、逐条预测与源码定位使这一流程可以复现和继续扩展。')

    # Appendix A: protocol and complete aggregate results.
    page(); h('附录 A  实验协议与完整分类结果')
    table(['项目', '本次设置'], [
        ['语料与标签', '人民日报 9801；按文章聚合段落，以版面号为标签。日期与原始编号用于组织样本，不进入正文特征。'],
        ['清洗与分词', f'去除拼音注释和格式标记，使用 jieba；去重 {stats["identical_duplicate_paragraphs_removed"]} 条重复段落，删除 {stats["short_articles_removed"]} 篇少于 30 个有效词的文章。'],
        ['日期划分', '训练：1 月 1–21 日；验证：22–26 日；测试：27–31 日。文章编号、日期和全文哈希跨集合互不重叠。'],
        ['表示学习范围', '词表、IDF 和表示只学习八个选定版面的训练日期文本；2/4/8 类共享这一无标签背景，各分类器使用本任务标签。'],
        ['选择与评估', '按验证宏 F1 选参，平分取网格中先出现的值；选定后评估测试集，保持训练集不变。每个配置 seed=42、单线程。'],
        ['类别权重', 'LR 使用 class_weight=balanced；原生监督式 fastText 使用普通样本损失。方法间比较对应各自完整训练路线。'],
    ], [91, 420])
    table(['类别数', '版面标签', '训练 / 验证 / 测试'], [
        [k, '、'.join(lookup[k, 'TFIDF']['labels']), ' / '.join(str(lookup[k, 'TFIDF']['sizes'][s]) for s in ['train', 'dev', 'test'])]
        for k in [2, 4, 8]
    ], [61, 278, 172])
    sub('全部 30 组测试宏 F1')
    methods = [r['method'] for r in result if r['classes'] == 8]
    table(['方法', '2 类', '4 类', '8 类'], [
        [names[m]] + [pct(lookup[k, m]['macro_f1']) for k in [2, 4, 8]] for m in methods
    ], [241, 90, 90, 90])
    p('宏 F1 对类别等权平均。2/4/8 类任务的标签集合与样本集合同步变化，各分数对应各自任务。中文分类中的 FastText 均值和加权基线使用 3–6 字符；1–3 字符模型由八类验证集选出后供三个任务使用，各任务重新选择 LR 的 C。', 'small')

    # Appendix B: diagnostics; measured errors, no unsupported causal attribution.
    page(); h('附录 B  类别分布与预测诊断')
    table(['8 类方法', 'Accuracy', 'Macro-F1', 'Weighted-F1'], [
        [names[m], pct(lookup[8, m]['accuracy']), pct(lookup[8, m]['macro_f1']), pct(lookup[8, m]['weighted_f1'])]
        for m in ['TFIDF', 'FastText_tuned', 'Supervised_fastText']
    ], [211, 100, 100, 100])
    table(['版面', '训练 / 验证 / 测试', 'TF-IDF F1', '1–3 FT F1', '监督 FT F1'], [
        [label, ' / '.join(str(stats[s]['pages'][label]) for s in ['train', 'dev', 'test'])] +
        [pct(lookup[8, m]['classification_report'][label]['f1-score']) for m in ['TFIDF', 'FastText_tuned', 'Supervised_fastText']]
        for label in sorted(stats['train']['pages'])
    ], [50, 155, 102, 102, 102])
    picture('fasttext_confusion_8.png', 440, 268,
            '图 2  1–3 字符 Gensim FastText 加权模型的八类测试混淆矩阵。数字为篇数，颜色按真实类别归一化。')
    p('主要混淆方向为：' + '；'.join(f'第 {q["gold"]} 版 → 第 {q["predicted"]} 版 {q["count"]} 篇' for q in audit['test_error_pairs'][:3]) + '。这些计数定位版面预测中的集中错误，逐篇文本与预测见保存的结果文件。', 'small')
    p('本任务以版面归属为预测目标，版面编号随日期对应不同报道安排。第 5、8 版测试支持度分别为 6、9 篇，按类指标应结合支持度阅读；日期划分检验的是同月后续日期上的泛化。正文结论限定于该单次实验及其明确的参数对照。', 'small')

    # Appendix C: implementation, surrounding code, and parameter meaning.
    page(); h('附录 C  源码对照与超参数含义')
    p('阅读版本为 Gensim 4.4.0 与 fastText 0.9.2。下表按“输入准备—训练调度—数值更新—推断保存”定位实现；完整函数行号与源码哈希保存在 SOURCE_NOTES.md 和 source_audit.json。', 'small')
    table(['代码位置', '论文机制或工程职责'], [
        ['word2vec.py / build_vocab、prepare_vocab', '统计词频、过滤低频词、设置高频词下采样；决定后续训练词表。'],
        ['word2vec.py / make_cum_table', '以词频的 ns_exponent 次幂构造负采样累计分布。'],
        ['word2vec.py / _job_producer、_worker_loop', '批次队列、学习率进度和线程私有缓冲；语料需可重复遍历。'],
        ['fasttext.py / _do_train_job → fasttext_inner.pyx / train_batch_any', 'Python 调度进入 Cython，准备动态窗口，在核心循环释放 GIL。'],
        ['fasttext_inner.pyx / fasttext_fast_sentence_sg_neg', '正负样本打分及整词、子词和上下文向量的原地更新。'],
        ['fasttext.py / ft_ngram_hashes、get_vector、adjust_vectors', '按 UTF-8 字符构造片段并取桶；组合词向量与 OOV 向量。'],
        ['dictionary.cc / addWordNgrams；model.cc / computeHidden、update', '监督分类的跨词特征、平均嵌入与联合梯度更新。'],
        ['fasttext.cc / createLoss；loss.cc', '选择 softmax、hs 或 ns；理解输出损失、Huffman 树和采样表。'],
    ], [231, 280])
    sub('参数选择同时关联数据、容量和训练过程')
    table(['参数', '设置与作用'], [
        ['dim、min_n / max_n、bucket', '50 维；比较 100 维。字符范围比较 3–6 / 1–3，桶数比较 20,000 / 2,000，控制表示容量与子词共享方式。'],
        ['window、min_count、sample', '5、2、0.001：分别控制上下文距离、整词最低词频和高频词下采样。'],
        ['sg、negative、ns_exponent', 'SG（另比较 CBOW）、5 个负样本、指数 0.75；Gensim 默认指数与原 fastText 的 0.5 分别记录。'],
        ['epochs、alpha、workers、seed', '中文词向量 10 轮；初始学习率 0.025；单线程、seed=42，PYTHONHASHSEED=42。'],
        ['LR C、原生 lr / wordNgrams', 'C∈{0.1,1,10}；原生 lr∈{0.1,0.3}、wordNgrams∈{1,2}，全部由验证宏 F1 选择。'],
        ['英文表示的固定设置', '词向量 20 轮；Doc2Vec PV-DBOW 训练/推断 30/30 轮；LSI 50 维；LDA 10 主题、5 passes。'],
    ], [151, 360])
    p('实现细节：Gensim 已登录向量对整词与子词之和除以“子词数+1”，OOV 对子词取均值；所读 SG 训练分支按 subwords_len 缩放。Gensim 的 word_ngrams 支持值为 1，原生 wordNgrams 控制跨词片段。OOV 统计使用 key_to_index，取向量时调用 get_vector；保存完整模型可保留子词合成能力。', 'small')

    # Appendix D: reproducibility and primary references.
    page(); h('附录 D  复现入口与参考资料')
    sub('运行与使用已训练模型')
    commands = [
        'python -m pip install -r requirements.txt',
        'python run_all.py',
        'python -m experiment.predict --method FastText_tuned --classes 8 --text "国际代表团就双边合作举行会谈。"',
        'python -m experiment.verify',
    ]
    for text in commands:
        story.append(Paragraph(escape(text), style['small']))
    md.append('```bash\n' + '\n'.join(commands) + '\n```\n')
    p('在项目根目录及独立 Python 环境中执行。先按 data/README.md 将语料放到 data/source_9801.txt.gz，运行 run_all.py 生成模型，再执行预测与核验。示例预测命令输出的是版面号。', 'small')
    p(f'已验证环境：Python {config["python"]}；Gensim {config["versions"]["gensim"]}；scikit-learn {config["versions"]["scikit-learn"]}；jieba {config["versions"]["jieba"]}；NumPy {config["versions"]["numpy"]}；SciPy {config["versions"]["scipy"]}；fasttext-wheel {config["versions"]["fasttext-wheel"]}。中文分类脚本本次耗时 {config["elapsed_seconds"]:.1f} 秒。', 'small')
    sub('结果与实现的对应文件')
    table(['文件', '内容'], [
        ['REPORT.md / 本 PDF', '同一构建脚本生成的报告文字、表格与图示。'],
        ['results/coverage.json、lee_representations.json', '中英文词次覆盖，英文八种表示、检索与主题结果。'],
        ['results/classification.json、fasttext_ablation.json', '完整分类指标、每类报告、混淆矩阵和参数对照。'],
        ['results/selection_grids.json、predictions.jsonl', '验证集候选选择与逐篇测试预测。'],
        ['SOURCE_NOTES.md、results/source_audit.json', '论文阅读、函数定位、参数解释和源码哈希。'],
        ['experiment/、models/、data/', '全流程实现；模型与中文语料保存在本地，按 README 准备并生成。'],
    ], [251, 260])
    p(f'模型重载后共 {sum(r["total"] for r in verify["reload_checks"]):,} 条预测与保存结果一致，覆盖八类的 9 种实际模型及 2/4 类原生 fastText。验证同时检查了划分隔离、正文清洗、词表冻结和 OOV 合成。复现入口统一设置 Python 哈希种子，以保持 Doc2Vec 推断初始化一致。', 'small')
    sub('参考资料（访问日期：2026-10-18）')
    refs = [
        ('Gensim 官网与 FastText API', 'https://radimrehurek.com/gensim/models/fasttext.html'),
        ('Gensim Lee / Doc2Vec 教程', 'https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html'),
        ('Bojanowski et al. (2017). Enriching Word Vectors with Subword Information.', 'https://aclanthology.org/Q17-1010/'),
        ('Joulin et al. (2017). Bag of Tricks for Efficient Text Classification.', 'https://aclanthology.org/E17-2068/'),
        ('Gensim 4.4.0 源码', 'https://github.com/piskvorky/gensim/tree/4.4.0/gensim/models'),
        ('fastText v0.9.2 源码', 'https://github.com/facebookresearch/fastText/tree/v0.9.2/src'),
        ('fastText 监督式分类教程', 'https://fasttext.cc/docs/en/supervised-tutorial.html'),
    ]
    for i, (label, url) in enumerate(refs, 1):
        story.append(Paragraph(f'[{i}] <link href="{escape(url)}" color="#147f86">{escape(label)}</link>', style['small']))
        md.append(f'{i}. [{label}]({url})')

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d3dfe4'))
        canvas.line(42, 39, 553, 39)
        canvas.setFont('CJK', 7.5)
        canvas.setFillColor(colors.HexColor('#647582'))
        canvas.drawString(42, 26, 'Gensim / FastText  ·  子词覆盖与中文适配')
        canvas.drawRightString(553, 26, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(ROOT / 'Gensim_FastText_Report.pdf'), pagesize=A4,
                            leftMargin=42, rightMargin=42, topMargin=39, bottomMargin=51,
                            title=title + '：Gensim 与 FastText 实验报告', author='')
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    (ROOT / 'REPORT.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    evidence = {
        'coverage_definition': 'finite nonzero vector / token occurrences',
        'english_token_coverage': {'Word2Vec': w2v[0] / 100, 'FastText': ft[0] / 100},
        'chinese_token_coverage': {'Word2Vec': w2v[1] / 100, 'FastText': ft[1] / 100},
        'short_word_type_fraction': short_fraction,
        'test_macro_f1_gain_percentage_points': gain,
        'source_files': ['classification.json', 'coverage.json', 'lee_representations.json',
                         'fasttext_ablation.json', 'source_audit.json', 'verification.json'],
    }
    (ROOT / 'results/report_evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(evidence, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--font', type=Path, required=True)
    build(parser.parse_args().font)
