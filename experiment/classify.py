"""Train-only features, validation-only selection, one held-out test per final method."""
import os
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import csv
import importlib.metadata
from pathlib import Path
import platform
import time

import fasttext
from gensim import matutils, models
import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .common import SEED, TASKS, read_rows, save_json
from .features import (bow_matrix, corpus_bow, dense_corpus, fit_dictionary, fit_doc2vec,
                       fit_embedding, infer_doc2vec, mean_vectors, positive_idf)

C_GRID = [.1, 1., 10.]
FT_CONFIGS = [
    {"name": "base_3_6", "dim": 50, "min_n": 3, "max_n": 6, "bucket": 20000, "sg": 1},
    {"name": "short_1_3", "dim": 50, "min_n": 1, "max_n": 3, "bucket": 20000, "sg": 1},
    {"name": "small_bucket", "dim": 50, "min_n": 3, "max_n": 6, "bucket": 2000, "sg": 1},
    {"name": "dim_100", "dim": 100, "min_n": 3, "max_n": 6, "bucket": 20000, "sg": 1},
    {"name": "cbow", "dim": 50, "min_n": 3, "max_n": 6, "bucket": 20000, "sg": 0},
]


def tune_lr(train_x, train_y, dev_x, dev_y):
    trials, best, best_score = [], None, -1.
    for c in C_GRID:
        model = LogisticRegression(C=c, class_weight="balanced", max_iter=1500,
                                   random_state=SEED, solver="lbfgs")
        started = time.perf_counter()
        model.fit(train_x, train_y)
        score = f1_score(dev_y, model.predict(dev_x), average="macro", zero_division=0)
        trials.append({"C": c, "dev_macro_f1": float(score),
                       "fit_seconds": time.perf_counter()-started,
                       "iterations": int(max(model.n_iter_))})
        if score > best_score:
            best, best_score = model, score
    return best, trials


def metrics(y, predicted, labels):
    return {"accuracy": float(accuracy_score(y, predicted)),
            "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, predicted, labels=labels, average="weighted", zero_division=0)),
            "classification_report": classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist()}


def native_predict(model, docs):
    return np.array([model.predict(" ".join(doc))[0][0].removeprefix("__label__") for doc in docs])


def run(data_dir, output, model_dir):
    started_all = time.perf_counter()
    data_dir, output, model_dir = map(Path, (data_dir, output, model_dir))
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    rows = {s: read_rows(data_dir/f"{s}.jsonl") for s in ("train", "dev", "test")}
    docs = {s: [r["tokens"] for r in rr] for s, rr in rows.items()}
    ys = {s: np.array([r["label"] for r in rr]) for s, rr in rows.items()}
    features, timings, coverage = {}, {}, {}
    np.random.seed(SEED)
    print("Fit dictionary / BoW / TFIDF / LSI", flush=True)
    start = time.perf_counter()
    dictionary = fit_dictionary(docs["train"])
    dictionary.save(str(model_dir/"dictionary.gensim"))
    bow = {s: corpus_bow(d, dictionary) for s, d in docs.items()}
    features["BoW"] = {s: normalize(bow_matrix(d, dictionary)) for s, d in docs.items()}
    timings["BoW"] = time.perf_counter()-start
    start = time.perf_counter()
    tfidf = models.TfidfModel(bow["train"], dictionary=dictionary, normalize=True)
    tfidf.save(str(model_dir/"tfidf.gensim"))
    tfidf_corpus = {s: list(tfidf[b]) for s, b in bow.items()}
    features["TFIDF"] = {s: matutils.corpus2csc(b, num_terms=len(dictionary)).T.tocsr() for s,b in tfidf_corpus.items()}
    timings["TFIDF"] = time.perf_counter()-start
    start = time.perf_counter()
    lsi = models.LsiModel(tfidf_corpus["train"], id2word=dictionary, num_topics=50, random_seed=SEED)
    lsi.save(str(model_dir/"lsi.gensim"))
    features["LSI"] = {s: normalize(dense_corpus(lsi[b],50)) for s,b in tfidf_corpus.items()}
    timings["LSI"] = time.perf_counter()-start

    print("Train Word2Vec SG", flush=True)
    start = time.perf_counter()
    w2v = fit_embedding(docs["train"], kind="word2vec")
    w2v.save(str(model_dir/"word2vec.gensim"), separately=[])
    features["Word2Vec_mean"], coverage["Word2Vec_mean"] = {}, {}
    for split in docs:
        features["Word2Vec_mean"][split], coverage["Word2Vec_mean"][split] = mean_vectors(docs[split], w2v.wv)
    timings["Word2Vec_mean"] = time.perf_counter()-start

    idf = positive_idf(docs["train"])
    joblib.dump(idf, model_dir/"positive_idf.joblib", compress=3)
    # All ablation choices use only the 8-class validation set.
    ablations, chosen, chosen_score, base = [], None, -1., None
    ft_train_times = {}
    for config in FT_CONFIGS:
        print("Train FastText", config, flush=True)
        start = time.perf_counter()
        ft = fit_embedding(docs["train"], **{k:v for k,v in config.items() if k!="name"})
        ft_train_times[config["name"]] = time.perf_counter()-start
        train_x, _ = mean_vectors(docs["train"], ft.wv, idf)
        dev_x, _ = mean_vectors(docs["dev"], ft.wv, idf)
        classifier, trials = tune_lr(train_x, ys["train"], dev_x, ys["dev"])
        score = max(t["dev_macro_f1"] for t in trials)
        ablations.append({**config, "epochs":10, "dev_macro_f1":score,
                          "best_C":float(classifier.C), "train_seconds":ft_train_times[config["name"]], "C_trials":trials})
        if config["name"] == "base_3_6":
            base = ft
        if score > chosen_score:
            chosen, chosen_score, chosen_config = ft, score, config
        save_json(output/"fasttext_ablation.json", ablations)
    base.save(str(model_dir/"fasttext_base.gensim"), separately=[])
    if chosen is not base:
        chosen.save(str(model_dir/"fasttext_tuned.gensim"), separately=[])
    tuned_path = "fasttext_base.gensim" if chosen is base else "fasttext_tuned.gensim"
    for name, model, weights in [("FastText_mean",base,None),("FastText_tfidf",base,idf),("FastText_tuned",chosen,idf)]:
        features[name], coverage[name] = {}, {}
        for split in docs:
            features[name][split], coverage[name][split] = mean_vectors(docs[split], model.wv, weights)
        timings[name] = ft_train_times["base_3_6" if name != "FastText_tuned" else chosen_config["name"]]
    print("Train Doc2Vec PV-DBOW; infer every split with the same procedure", flush=True)
    start = time.perf_counter()
    d2v = fit_doc2vec(docs["train"])
    features["Doc2Vec"] = {s: infer_doc2vec(d2v, d) for s,d in docs.items()}
    d2v.save(str(model_dir/"doc2vec.gensim"), separately=[])
    timings["Doc2Vec"] = time.perf_counter()-start
    save_json(output/"coverage.json", coverage)
    save_json(output/"representation_seconds.json", timings)

    records, grids, predictions = [], {}, []
    for k, labels in TASKS.items():
        mask = {s: np.isin(y, labels) for s,y in ys.items()}
        task_y = {s: y[mask[s]] for s,y in ys.items()}
        task_docs = {s: [r["tokens"] for r in rows[s] if r["label"] in labels] for s in rows}
        test_rows = [r for r in rows["test"] if r["label"] in labels]
        sizes = {s:len(task_y[s]) for s in task_y}
        def record(name, predicted, dev_score, seconds, params):
            result = {"classes":k,"method":name,"labels":labels,"sizes":sizes,
                      "dev_macro_f1":float(dev_score), "classifier_selection_seconds":seconds,
                      "parameters":params, **metrics(task_y["test"],predicted,labels)}
            records.append(result)
            predictions.extend({"classes":k,"method":name,"id":r["id"],"gold":r["label"],"predicted":str(p)} for r,p in zip(test_rows,predicted))
            print(k,name,"dev",round(dev_score,4),"test",round(result["macro_f1"],4),flush=True)
            save_json(output/"classification.json",records)
        majority = DummyClassifier(strategy="most_frequent").fit(np.zeros((sizes["train"],1)),task_y["train"])
        p = majority.predict(np.zeros((sizes["test"],1)))
        d = majority.predict(np.zeros((sizes["dev"],1)))
        record("Majority",p,f1_score(task_y["dev"],d,average="macro",zero_division=0),0.,{})
        for name, vectors in features.items():
            start = time.perf_counter()
            classifier, trials = tune_lr(vectors["train"][mask["train"]],task_y["train"],vectors["dev"][mask["dev"]],task_y["dev"])
            elapsed = time.perf_counter()-start
            grids[f"{k}_{name}"] = trials
            predicted = classifier.predict(vectors["test"][mask["test"]])
            joblib.dump(classifier,model_dir/f"classifier_{k}_{name}.joblib",compress=3)
            record(name,predicted,max(t["dev_macro_f1"] for t in trials),elapsed,{"C":float(classifier.C),"class_weight":"balanced"})

        print("Tune native supervised fastText",k,flush=True)
        train_file = data_dir/f"fasttext_train_{k}.txt"
        train_file.write_text("\n".join(f"__label__{y} "+" ".join(d) for y,d in zip(task_y["train"],task_docs["train"]))+"\n",encoding="utf-8")
        trials, best, best_score, start = [], None, -1., time.perf_counter()
        for word_ngrams in (1,2):
            for lr in (.1,.3):
                model = fasttext.train_supervised(input=str(train_file),epoch=25,lr=lr,dim=50,
                    minCount=2,wordNgrams=word_ngrams,minn=0,maxn=0,bucket=20000,thread=1,
                    seed=SEED,loss="softmax",verbose=0)
                score = f1_score(task_y["dev"],native_predict(model,task_docs["dev"]),average="macro",zero_division=0)
                trials.append({"wordNgrams":word_ngrams,"lr":lr,"dev_macro_f1":float(score)})
                if score > best_score:
                    best,best_score,best_parameters=model,score,{"wordNgrams":word_ngrams,"lr":lr}
        elapsed=time.perf_counter()-start
        best.save_model(str(model_dir/f"native_fasttext_{k}.bin"))
        grids[f"{k}_Supervised_fastText"]=trials
        record("Supervised_fastText",native_predict(best,task_docs["test"]),best_score,elapsed,
               {**best_parameters,"epoch":25,"dim":50,"minCount":2,"minn":0,"maxn":0,"bucket":20000,"loss":"softmax","thread":1,"seed":SEED})

    save_json(output/"selection_grids.json",grids)
    with (output/"predictions.jsonl").open("w",encoding="utf-8") as f:
        import json
        for row in predictions: f.write(json.dumps(row,ensure_ascii=False)+"\n")
    with (output/"classification.csv").open("w",encoding="utf-8-sig",newline="") as f:
        fields=["classes","method","dev_macro_f1","accuracy","macro_f1","weighted_f1","classifier_selection_seconds"]
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        writer.writerows({key:r[key] for key in fields} for r in records)
    config={"seed":SEED,"python_hash_seed":os.environ.get("PYTHONHASHSEED"),"fasttext_tuned":chosen_config,"fasttext_tuned_path":tuned_path,
            "dictionary_size":len(dictionary),"word_vocabulary_size":len(base.wv),
            "feature_scope":"all 8 classes, training dates only; supervised labels restricted to each task",
            "selection_rule":"highest validation macro-F1; first grid item wins exact tie; no train+dev refit",
            "C_grid":C_GRID,"word_embeddings":{"dim":50,"window":5,"min_count":2,"epochs":10,"sg":1,"negative":5,"ns_exponent":.75,"sample":.001,"workers":1},
            "Doc2Vec":{"dm":0,"dbow_words":1,"train_epochs":20,"infer_epochs":30,"dim":50},
            "elapsed_seconds":time.perf_counter()-started_all,"platform":platform.platform(),"python":platform.python_version(),
            "versions":{p:importlib.metadata.version(p) for p in ["gensim","numpy","scipy","scikit-learn","jieba","fasttext-wheel","joblib"]}}
    save_json(output/"run_config.json",config)
    print("COMPLETE",config["elapsed_seconds"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--data",default="data/processed")
    parser.add_argument("--output",default="results")
    parser.add_argument("--models",default="models")
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.data,args.output,args.models)
