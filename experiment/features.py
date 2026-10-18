"""Gensim representations; fit only on the explicitly supplied training texts."""
from collections import Counter
import hashlib
import time

import numpy as np
from gensim import corpora, matutils, models
from gensim.models.doc2vec import TaggedDocument
from sklearn.preprocessing import normalize

from .common import SEED, stable_hash


def dense_corpus(corpus, dimensions):
    return matutils.corpus2dense(corpus, num_terms=dimensions).T.astype(np.float32)


def fit_dictionary(documents, no_below=2, keep_n=30000):
    dictionary = corpora.Dictionary(documents)
    dictionary.filter_extremes(no_below=no_below, no_above=.95, keep_n=keep_n)
    dictionary.compactify()
    return dictionary


def bow_matrix(documents, dictionary):
    bows = [dictionary.doc2bow(doc, allow_update=False) for doc in documents]
    return matutils.corpus2csc(bows, num_terms=len(dictionary), dtype=np.float32).T.tocsr()


def corpus_bow(documents, dictionary):
    return [dictionary.doc2bow(doc, allow_update=False) for doc in documents]


def fit_embedding(documents, kind="fasttext", dim=50, epochs=10, min_n=3, max_n=6, bucket=20000, sg=1):
    cls = models.FastText if kind == "fasttext" else models.Word2Vec
    extra = {"min_n": min_n, "max_n": max_n, "bucket": bucket} if kind == "fasttext" else {}
    model = cls(vector_size=dim, window=5, min_count=2, sg=sg, negative=5, hs=0,
                sample=1e-3, ns_exponent=.75, alpha=.025, min_alpha=.0001,
                workers=1, seed=SEED, hashfxn=stable_hash, shrink_windows=True, **extra)
    model.build_vocab(documents)
    model.train(documents, total_examples=model.corpus_count, epochs=epochs)
    return model


def mean_vectors(documents, keyed_vectors, idf=None):
    """TF * IDF weighted mean, with a positive fallback IDF for FastText OOV.

    FastText membership is not vocabulary membership. get_vector may synthesize
    OOV vectors, so do not discard a token just because it is absent from
    key_to_index. A truly unrepresentable document gets a finite zero vector.
    """
    matrix = np.zeros((len(documents), keyed_vectors.vector_size), dtype=np.float32)
    stats = {"tokens": 0, "oov_tokens": 0, "represented_tokens": 0, "zero_documents": 0}
    fallback_idf = max(idf.values(), default=1.) if idf is not None else 1.
    for i, doc in enumerate(documents):
        denominator = 0.
        for token, count in Counter(doc).items():
            stats["tokens"] += count
            if token not in keyed_vectors.key_to_index:
                stats["oov_tokens"] += count
            try:
                vector = keyed_vectors.get_vector(token)
            except KeyError:
                continue
            if not np.isfinite(vector).all() or np.linalg.norm(vector) == 0:
                continue
            weight = count * (idf.get(token, fallback_idf) if idf is not None else 1.)
            matrix[i] += vector * weight
            denominator += weight
            stats["represented_tokens"] += count
        if denominator:
            matrix[i] /= denominator
        else:
            stats["zero_documents"] += 1
    return normalize(matrix), stats


def fit_doc2vec(documents, dim=50, epochs=20):
    tagged = [TaggedDocument(doc, [i]) for i, doc in enumerate(documents)]
    model = models.Doc2Vec(vector_size=dim, dm=0, dbow_words=1, window=5, min_count=2,
                          epochs=epochs, workers=1, seed=SEED, hashfxn=stable_hash,
                          negative=5, hs=0, sample=1e-3)
    model.build_vocab(tagged)
    model.train(tagged, total_examples=model.corpus_count, epochs=epochs)
    return model


def infer_doc2vec(model, documents, epochs=30):
    result = []
    for doc in documents:
        seed = int(hashlib.sha256(" ".join(doc).encode()).hexdigest()[:8], 16)
        model.random.seed(seed)
        result.append(model.infer_vector(doc, epochs=epochs, alpha=.025, min_alpha=.0001))
    return normalize(np.asarray(result, dtype=np.float32))


def positive_idf(documents):
    counts = Counter(token for doc in documents for token in set(doc))
    return {token: float(np.log((1+len(documents))/(1+df)) + 1.) for token, df in counts.items()}

