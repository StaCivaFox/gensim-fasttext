"""Clean the user-supplied 9801 corpus; one sample is a whole article."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import unicodedata

import jieba

from .common import PAGE_ORDER, TASKS, save_json, sha256

LINE = re.compile(r"^(\d{8})-(\d{2})-(\d{3})-(\d{3})\s+(.*)$")
PINYIN = re.compile(r"\{[^{}]*\}")
PICTURE = re.compile(r"[（(]附?图片\s*\d*\s*张?[)）]")
TOKEN = re.compile(r"^[\u3400-\u9fffA-Za-z0-9]+$")
# Small explicit function-word list, fixed before examining labels or scores.
STOPWORDS = set("的 了 和 是 在 对 与 及 或 而 也 有 为 以 于 把 被 将 等 地 得 着 这 那 一个 一种 进行 通过 以及 但是 因为 所以 这个 那个 我们 你们 他们 她们 它们 其 之 并 并且 到 从 向 中 上 下 后 前 本 次 个 年 月 日 说 表示".split())
SEGMENTER = jieba.Tokenizer()


def clean_text(text):
    text = unicodedata.normalize("NFKC", PINYIN.sub("", text))
    # Corpus spaces are segmentation annotations, not natural English spaces.
    text = "".join(text.split())
    return PICTURE.sub("", text).replace("〓", "")


def tokenize(text):
    tokens = []
    for token in SEGMENTER.cut(text, HMM=False):
        token = token.strip().lower()
        if not token or token in STOPWORDS or not TOKEN.fullmatch(token):
            continue
        if token.isdigit():
            token = "numbertoken"
        tokens.append(token)
    return tokens


def split_for_date(date):
    day = int(date[-2:])
    return "train" if day <= 21 else "dev" if day <= 26 else "test"


def prepare(raw_path, output):
    raw_path, output = Path(raw_path), Path(output)
    raw = gzip.decompress(raw_path.read_bytes()) if raw_path.suffix == ".gz" else raw_path.read_bytes()
    try:
        content, encoding = raw.decode("utf-8-sig"), "utf-8-sig"
    except UnicodeDecodeError:
        content, encoding = raw.decode("gb18030"), "gb18030"
    articles = defaultdict(dict)
    ignored = 0
    repeated_paragraphs = 0
    for line in content.splitlines():
        if not line.strip():
            continue
        match = LINE.match(line)
        if not match:
            ignored += 1
            continue
        date, page, article, paragraph, body = match.groups()
        if not date.startswith("199801") or page not in PAGE_ORDER:
            continue
        key = f"{date}-{page}-{article}"
        cleaned = clean_text(body)
        if paragraph in articles[key]:
            if articles[key][paragraph] != cleaned:
                raise ValueError(f"Conflicting paragraph ID {key}-{paragraph}; inspect source before proceeding")
            repeated_paragraphs += 1
            continue
        articles[key][paragraph] = cleaned
    rows = {name: [] for name in ["train", "dev", "test"]}
    seen = set()
    short, duplicates = 0, 0
    for article_id, paragraphs in sorted(articles.items()):
        text = "\n".join(paragraphs[key] for key in sorted(paragraphs))
        fingerprint = hashlib.sha256("".join(text.split()).encode()).hexdigest()
        if fingerprint in seen:
            duplicates += 1
            continue
        tokens = tokenize(text)
        if len(tokens) < 30:
            short += 1
            continue
        seen.add(fingerprint)
        date, page, _ = article_id.split("-")
        row = {"id": article_id, "date": date, "label": page, "text": text,
               "tokens": tokens, "paragraphs": len(paragraphs), "sha256": fingerprint}
        rows[split_for_date(date)].append(row)
    output.mkdir(parents=True, exist_ok=True)
    stats = {"source": raw_path.name, "source_file_sha256": sha256(raw_path),
             "uncompressed_sha256": hashlib.sha256(raw).hexdigest(), "encoding": encoding,
             "label_type": "newspaper_page_id_not_curated_topic", "page_order": PAGE_ORDER,
             "tasks": TASKS, "ignored_nonempty_lines": ignored, "short_articles_removed": short,
             "identical_duplicate_paragraphs_removed": repeated_paragraphs,
             "duplicate_articles_removed": duplicates, "minimum_tokens": 30,
             "split_rule": "train: Jan 01-21; dev: Jan 22-26; test: Jan 27-31",
             "segmentation": "jieba 0.42.1, HMM=False, fixed function words, digits -> numbertoken"}
    for split, values in rows.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for row in values:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
        stats[split] = {"articles": len(values), "tokens": sum(len(r["tokens"]) for r in values),
                        "characters": sum(len(r["text"]) for r in values), "pages": dict(sorted(Counter(r["label"] for r in values).items())),
                        "dates": sorted({r["date"] for r in values})}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        for field in ["id", "date", "sha256"]:
            assert not ({r[field] for r in rows[a]} & {r[field] for r in rows[b]})
    for split in rows:
        assert set(stats[split]["pages"]) == set(PAGE_ORDER), f"Missing class in {split}"
    save_json(output / "stats.json", stats)
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=Path("data/source_9801.txt.gz"))
    parser.add_argument("--output", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.output), ensure_ascii=False, indent=2))

