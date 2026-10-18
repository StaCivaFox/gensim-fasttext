"""Record source locations, versions, subword hashing and observable corpus diagnostics."""
import argparse
from collections import Counter
import inspect
from pathlib import Path

import gensim
from gensim.models import FastText
from gensim.models.fasttext_inner import compute_ngrams_bytes,ft_hash_bytes

from .common import load_json,read_rows,save_json,sha256


def run(native_source=None):
    package=Path(gensim.__file__).parent
    files=["models/fasttext.py","models/fasttext_inner.pyx","models/word2vec.py","models/doc2vec.py"]
    sources=[]
    for f in files:
        sources.append({"path":"gensim/"+f,"sha256":sha256(package/f),
                        "url":"https://github.com/piskvorky/gensim/blob/4.4.0/gensim/"+f})
    if native_source:
        root=Path(native_source)
        for f in ["src/dictionary.cc","src/model.cc","src/fasttext.cc","src/loss.cc","python/fasttext_module/fasttext/FastText.py"]:
            sources.append({"path":"fasttext-0.9.2/"+f,"sha256":sha256(root/f),
                            "url":"https://github.com/facebookresearch/fastText/blob/v0.9.2/"+f})
    model=FastText.load("models/fasttext_base.gensim")
    vocab=model.wv.index_to_key
    collisions=[]
    for minn,maxn,bucket in [(3,6,20000),(1,3,20000),(3,6,2000)]:
        ngrams={n for word in vocab for n in compute_ngrams_bytes(word,minn,maxn)}
        counts=Counter(ft_hash_bytes(n)%bucket for n in ngrams)
        collisions.append({"min_n":minn,"max_n":maxn,"bucket":bucket,
            "unique_character_ngrams":len(ngrams),"occupied_buckets":len(counts),
            "collision_fraction":1-len(counts)/len(ngrams),"largest_bucket_distinct_ngrams":max(counts.values()),
            "ngram_matrix_mib_float32_dim50":bucket*50*4/1024**2})
    rows={s:read_rows(f"data/processed/{s}.jsonl") for s in ["train","dev","test"]}
    token_lengths=Counter(len(w) for w in vocab)
    predictions=read_rows("results/predictions.jsonl")
    subset=[p for p in predictions if p["classes"]==8 and p["method"]=="FastText_tuned"]
    pairs=Counter((p["gold"],p["predicted"]) for p in subset if p["gold"]!=p["predicted"])
    row_lookup={r["id"]:r for r in rows["test"]}
    examples=[{**p,"title":row_lookup[p["id"]]["text"].splitlines()[0][:60]} for p in subset if p["gold"]!=p["predicted"]][:10]
    report={"gensim_version":gensim.__version__,"fasttext_signature":str(inspect.signature(FastText)),
            "source_files":sources,"hash_collisions":collisions,"character_examples":{w:[n.decode("utf-8") for n in compute_ngrams_bytes(w,3,6)] for w in ["中国","经济","体育新闻"]},
            "vocabulary_word_lengths":dict(sorted(token_lengths.items())),"max_article_tokens":{s:max(len(r["tokens"]) for r in rr) for s,rr in rows.items()},
            "test_error_pairs":[{"gold":a,"predicted":b,"count":n} for (a,b),n in pairs.most_common(10)],"test_error_examples":examples,
            "notes":"Collision fraction is 1 - occupied buckets / distinct ngram strings. It is not an error rate. Only retained training vocabulary is scanned."}
    save_json("results/source_audit.json",report)
    print("source and hash audit complete")


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--native-source",help="Optional local unpacked official fasttext-0.9.2 PyPI sdist")
    run(p.parse_args().native_source)
