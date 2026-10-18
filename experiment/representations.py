"""Eight Gensim representations on the packaged Lee news corpus (not mixed with 9801)."""
import os
for _name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):
    os.environ.setdefault(_name,"1")
from pathlib import Path
import time

import gensim
from gensim import matutils, models
from gensim.parsing.preprocessing import STOPWORDS
from gensim.utils import simple_preprocess
import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

from .common import SEED, save_json, sha256
from .features import (bow_matrix, corpus_bow, dense_corpus, fit_dictionary, fit_doc2vec,
                       fit_embedding, infer_doc2vec, mean_vectors)


def preprocess(text):
    return [w for w in simple_preprocess(text,deacc=True) if w not in STOPWORDS]


def run(output="results/lee_representations.json"):
    np.random.seed(SEED)
    folder=Path(gensim.__file__).parent/"test/test_data"
    paths={"train":folder/"lee_background.cor","test":folder/"lee.cor"}
    docs={s:[preprocess(t) for t in p.read_text(encoding="iso-8859-1").splitlines() if t.strip()] for s,p in paths.items()}
    queries=["government election parliament","bushfire firefighters evacuation","cricket match team"]
    query_docs=[preprocess(q) for q in queries]
    all_docs={**docs,"query":query_docs}
    reps, summary, nearest={},[],{}
    dictionary=fit_dictionary(docs["train"])
    bow={s:corpus_bow(d,dictionary) for s,d in all_docs.items()}
    t=time.perf_counter()
    reps["BoW"]={s:normalize(bow_matrix(d,dictionary)) for s,d in all_docs.items()}
    times={"BoW":time.perf_counter()-t}
    t=time.perf_counter()
    tfidf=models.TfidfModel(bow["train"],dictionary=dictionary,normalize=True)
    tfcorpus={s:list(tfidf[b]) for s,b in bow.items()}
    reps["TFIDF"]={s:matutils.corpus2csc(b,num_terms=len(dictionary)).T.tocsr() for s,b in tfcorpus.items()}
    times["TFIDF"]=time.perf_counter()-t
    t=time.perf_counter()
    lsi=models.LsiModel(tfcorpus["train"],id2word=dictionary,num_topics=50,random_seed=SEED)
    reps["LSI"]={s:normalize(dense_corpus(lsi[b],50)) for s,b in tfcorpus.items()}
    times["LSI"]=time.perf_counter()-t
    t=time.perf_counter()
    lda=models.LdaModel(bow["train"],id2word=dictionary,num_topics=10,passes=5,iterations=50,
                        random_state=SEED,alpha="symmetric",eta="symmetric",minimum_probability=0.)
    reps["LDA"]={s:dense_corpus(lda[b],10) for s,b in bow.items()}
    times["LDA"]=time.perf_counter()-t
    embedding_models={}
    coverage={}
    for name,kind,sg in [("Word2Vec_CBOW","word2vec",0),("Word2Vec_SG","word2vec",1),("FastText","fasttext",1)]:
        print("Lee",name,flush=True)
        t=time.perf_counter()
        model=fit_embedding(docs["train"],kind=kind,sg=sg,epochs=20)
        embedding_models[name]=model
        reps[name],coverage[name]={},{}
        for split in all_docs:
            reps[name][split],coverage[name][split]=mean_vectors(all_docs[split],model.wv)
        times[name]=time.perf_counter()-t
        nearest[name]={w:[[n,float(v)] for n,v in model.wv.most_similar(w,topn=5)] for w in ["government","fire","cricket"] if w in model.wv.key_to_index}
    t=time.perf_counter()
    d2v=fit_doc2vec(docs["train"],epochs=30)
    reps["Doc2Vec"]={s:infer_doc2vec(d2v,d) for s,d in all_docs.items()}
    times["Doc2Vec"]=time.perf_counter()-t
    for name,x in reps.items():
        matrix=x["train"]
        nnz=matrix.nnz if sparse.issparse(matrix) else np.count_nonzero(matrix)
        summary.append({"method":name,"train_shape":list(matrix.shape),"test_shape":list(x["test"].shape),
                        "nonzero_fraction":float(nnz/np.prod(matrix.shape)),"fit_and_transform_seconds":times[name]})
    retrieval=[]
    for name,x in reps.items():
        scores=cosine_similarity(x["query"],x["train"])
        for i,query in enumerate(queries):
            rank=np.argsort(-scores[i],kind="stable")[:3]
            retrieval.append({"method":name,"query":query,"top3":[{"train_doc_id":int(j),"cosine":float(scores[i,j])} for j in rank]})
    ft=embedding_models["FastText"].wv
    wv=embedding_models["Word2Vec_SG"].wv
    unseen=sorted({w for d in docs["test"] for w in d if w not in wv.key_to_index})
    synthetic=[w for w in ["firefighterx","parliamentarianx","cricketmatchx"] if w not in wv.key_to_index]
    oov=[]
    for token in unseen[:8]+synthetic:
        vector=ft.get_vector(token)
        oov.append({"token":token,"synthetic":token in synthetic,"Word2Vec_in_vocabulary":token in wv.key_to_index,
                    "FastText_in_vocabulary":token in ft.key_to_index,"FastText_norm":float(np.linalg.norm(vector)),
                    "FastText_nearest":[[n,float(v)] for n,v in ft.most_similar(token,topn=3)]})
    result={"gensim_version":gensim.__version__,"source_url":"https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html",
            "sources":{s:{"package_path":f"gensim/test/test_data/{p.name}","sha256":sha256(p)} for s,p in paths.items()},
            "documents":{s:len(d) for s,d in docs.items()},"tokens":{s:sum(map(len,d)) for s,d in docs.items()},
            "dictionary_size":len(dictionary),"preprocessing":"simple_preprocess(deacc=True), Gensim English STOPWORDS",
            "parameters":{"seed":SEED,"dimensions":50,"embedding_epochs":20,"embedding_window":5,"min_count":2,
                          "FastText":{"min_n":3,"max_n":6,"bucket":20000,"sg":1},
                          "Doc2Vec":{"dm":0,"dbow_words":1,"train_epochs":30,"infer_epochs":30},
                          "LDA":{"topics":10,"passes":5,"iterations":50}},
            "summary":summary,"coverage":coverage,"nearest_words":nearest,"retrieval":retrieval,
            "topics":[{"topic":i,"words":[[w,float(p)] for w,p in lda.show_topic(i,topn=8)]} for i in range(10)],
            "oov_examples":oov,"evaluation_note":"Qualitative exploration; no relevance labels, so no retrieval accuracy claim. Training documents are indexed from zero."}
    save_json(output,result)
    print("Lee complete",result["documents"],flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):
        run()
