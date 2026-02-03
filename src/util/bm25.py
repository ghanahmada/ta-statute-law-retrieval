import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class BM25(object):
    def __init__(self, b=0.7, k1=1.6, n_gram:int = 1):
        self.n_gram = n_gram
        self.vectorizer = TfidfVectorizer(max_df=.65, min_df=1,
                                  use_idf=True, 
                                  ngram_range=(n_gram, n_gram))
        
        self.b = b
        self.k1 = k1
        self.len_X = 0
        self.X_csc = None

    def fit(self, X):
        """ Fit IDF to documents X """
        start_time = time.perf_counter()
        print(f"Fitting tf_idf vectorizer")
        self.vectorizer = self.vectorizer.fit(X)
        mid_time = time.perf_counter()
        print(f"Finished tf_idf vectorizer, time : {mid_time - start_time:0.3f} sec")
        y = super(TfidfVectorizer, self.vectorizer).transform(X)
        end_time = time.perf_counter()
        print(f"Finished corpus transform, time : {end_time - mid_time:0.3f} sec")
        self.avdl = y.sum(1).mean()
        self.len_X = y.sum(1).A1
        self.X_csc = y.tocsc()
        final_time = time.perf_counter()
        print(f"Finished sparsifying, time: {final_time - end_time:0.3f} sec")
        print()

    def transform(self, q):
        """ Calculate BM25 between query q and documents X """
        b, k1, avdl = self.b, self.k1, self.avdl

        # apply CountVectorizer
        len_X = self.len_X
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        # assert sparse.isspmatrix_csr(q)

        # convert to csc for better column slicing
        X = self.X_csc[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1