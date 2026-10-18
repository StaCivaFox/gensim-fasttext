"""Meaningful leakage and model-reload checks against saved held-out predictions."""
import os
for key in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(key,"1")
import time
import numpy as np
from gensim import corpora,models
from threadpoolctl import threadpool_limits

from .common import read_rows,save_json,TASKS
from .features import mean_vectors
from .predict import Predictor
from .prepare import clean_text,tokenize


def run():
    checks=[]
    start=time.perf_counter()
    rows={s:read_rows(f"data/processed/{s}.jsonl") for s in ["train","dev","test"]}
    for a,b in [("train","dev"),("train","test"),("dev","test")]:
        for field in ["id","date","sha256"]:
            assert not ({r[field] for r in rows[a]} & {r[field] for r in rows[b]})
    checks.append("PASS: article IDs, dates and exact text hashes do not overlap across splits")
    assert clean_text("经{jing1} 济 （ 附 图片 １ 张 ）") == "经济"
    assert "19980101-01-001" not in tokenize(rows["train"][0]["text"])
    assert all(not t.startswith("199801") for r in rows["train"] for t in r["tokens"])
    checks.append("PASS: pinyin, picture markers and source metadata are excluded from features")
    dictionary=corpora.Dictionary.load("models/dictionary.gensim")
    before=dict(dictionary.token2id)
    train_words={t for r in rows["train"] for t in r["tokens"]}
    assert set(before)<=train_words
    for r in rows["test"]: dictionary.doc2bow(r["tokens"],allow_update=False)
    assert dictionary.token2id==before
    checks.append("PASS: dictionary only contains training terms and remains unchanged on test transform")
    ft=models.FastText.load("models/fasttext_base.gensim")
    token="新闻词向量专用未登录测试词xyz"
    assert token not in ft.wv.key_to_index
    vector,stats=mean_vectors([[token]],ft.wv)
    assert stats["represented_tokens"]==1 and np.isfinite(vector).all() and np.linalg.norm(vector)>0
    checks.append("PASS: an OOV token is composed by FastText instead of silently discarded")
    saved=read_rows("results/predictions.jsonl")
    matches=[]
    methods=["BoW","TFIDF","LSI","Word2Vec_mean","FastText_mean","FastText_tfidf","FastText_tuned","Doc2Vec","Supervised_fastText"]
    for method in methods:
        actual=Predictor(method,8).predict_tokens([r["tokens"] for r in rows["test"]])
        reference={r["id"]:r["predicted"] for r in saved if r["classes"]==8 and r["method"]==method}
        assert actual==[reference[r["id"]] for r in rows["test"]],method
        matches.append({"method":method,"classes":8,"matched":len(actual),"total":len(actual)})
    for k in [2,4]:
        selected=[r for r in rows["test"] if r["label"] in TASKS[k]]
        actual=Predictor("Supervised_fastText",k).predict_tokens([r["tokens"] for r in selected])
        reference={r["id"]:r["predicted"] for r in saved if r["classes"]==k and r["method"]=="Supervised_fastText"}
        assert actual==[reference[r["id"]] for r in selected]
        matches.append({"method":"Supervised_fastText","classes":k,"matched":len(actual),"total":len(actual)})
    checks.append("PASS: reloaded feature models and classifiers reproduce stored held-out predictions")
    report={"checks":checks,"reload_checks":matches,"elapsed_seconds":time.perf_counter()-start}
    save_json("results/verification.json",report)
    print("\n".join(checks))


if __name__=="__main__":
    # infer_vector's initializer uses Python's process hash: set it at startup,
    # not after the interpreter has already chosen its hash secret.
    if os.environ.get("PYTHONHASHSEED") != "42":
        import subprocess,sys
        subprocess.run([sys.executable,"-m","experiment.verify"],
                       env=dict(os.environ,PYTHONHASHSEED="42"),check=True)
    else:
        with threadpool_limits(limits=1):run()
