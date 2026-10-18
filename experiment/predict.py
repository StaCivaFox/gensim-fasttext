"""Reload saved models and classify new Chinese text into newspaper page IDs."""
import argparse
import json
from pathlib import Path

import fasttext
from gensim import corpora, matutils, models
import joblib
from sklearn.preprocessing import normalize

from .common import load_json
from .features import bow_matrix, corpus_bow, dense_corpus, infer_doc2vec, mean_vectors
from .prepare import clean_text, tokenize


class Predictor:
    def __init__(self, method="FastText_tuned", classes=8, model_dir="models", results_dir="results"):
        self.method=method
        self.folder=Path(model_dir)
        self.config=load_json(Path(results_dir)/"run_config.json")
        if method=="Supervised_fastText":
            self.model=fasttext.load_model(str(self.folder/f"native_fasttext_{classes}.bin"))
            return
        self.classifier=joblib.load(self.folder/f"classifier_{classes}_{method}.joblib")
        if method in ["BoW","TFIDF","LSI"]:
            self.dictionary=corpora.Dictionary.load(str(self.folder/"dictionary.gensim"))
            if method!="BoW": self.tfidf=models.TfidfModel.load(str(self.folder/"tfidf.gensim"))
            if method=="LSI": self.lsi=models.LsiModel.load(str(self.folder/"lsi.gensim"))
        elif method=="Word2Vec_mean":
            self.model=models.Word2Vec.load(str(self.folder/"word2vec.gensim"))
        elif method.startswith("FastText"):
            path=self.config["fasttext_tuned_path"] if method=="FastText_tuned" else "fasttext_base.gensim"
            self.model=models.FastText.load(str(self.folder/path))
            self.idf=None if method=="FastText_mean" else joblib.load(self.folder/"positive_idf.joblib")
        elif method=="Doc2Vec":
            self.model=models.Doc2Vec.load(str(self.folder/"doc2vec.gensim"))
        else:
            raise ValueError(f"Unknown method: {method}")

    def predict_tokens(self, documents):
        if self.method=="Supervised_fastText":
            return [self.model.predict(" ".join(d))[0][0].removeprefix("__label__") for d in documents]
        if self.method=="BoW":
            x=normalize(bow_matrix(documents,self.dictionary))
        elif self.method in ["TFIDF","LSI"]:
            corpus=list(self.tfidf[corpus_bow(documents,self.dictionary)])
            x=(matutils.corpus2csc(corpus,num_terms=len(self.dictionary)).T.tocsr() if self.method=="TFIDF"
               else normalize(dense_corpus(self.lsi[corpus],50)))
        elif self.method=="Doc2Vec":
            x=infer_doc2vec(self.model,documents)
        else:
            x,_=mean_vectors(documents,self.model.wv,getattr(self,"idf",None))
        return self.classifier.predict(x).tolist()

    def predict_texts(self,texts):
        return self.predict_tokens([tokenize(clean_text(t)) for t in texts])


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--text",required=True)
    parser.add_argument("--method",default="FastText_tuned")
    parser.add_argument("--classes",type=int,choices=[2,4,8],default=8)
    args=parser.parse_args()
    label=Predictor(args.method,args.classes).predict_texts([args.text])[0]
    print(json.dumps({"page_id":label,"method":args.method,"classes":args.classes},ensure_ascii=False))
