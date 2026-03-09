import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _build_indonesian_preprocessor(use_stemmer: bool, use_stopwords: bool):
    """Build a text preprocessor for Indonesian using PySastrawi."""
    stemmer_fn = None
    stop_words: set = set()

    if use_stemmer:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
        stemmer_fn = StemmerFactory().create_stemmer().stem

    if use_stopwords:
        from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
        stop_words = set(StopWordRemoverFactory().get_stop_words())

    def preprocess(text: str) -> str:
        words = text.lower().split()
        if stop_words:
            words = [w for w in words if w not in stop_words]
        if stemmer_fn:
            words = [stemmer_fn(w) for w in words]
        return " ".join(words)

    return preprocess


class BM25(object):
    def __init__(self, b=0.7, k1=1.6, n_gram: int = 1,
                 lang: str = "en", use_stemmer: bool = False, use_stopwords: bool = False):
        self.n_gram = n_gram
        self.lang = lang
        self.use_stemmer = use_stemmer
        self.use_stopwords = use_stopwords
        self._preprocessor = None

        if lang == "id" and (use_stemmer or use_stopwords):
            self._preprocessor = _build_indonesian_preprocessor(use_stemmer, use_stopwords)

        self.vectorizer = TfidfVectorizer(
            max_df=.65, min_df=1,
            use_idf=True,
            ngram_range=(n_gram, n_gram),
        )
        self.b = b
        self.k1 = k1
        self.len_X = 0
        self.X_csc = None
        self.avdl = 0

    def _preprocess_corpus(self, texts):
        if self._preprocessor is None:
            return texts
        return [self._preprocessor(t) for t in texts]

    def _preprocess_query(self, query: str) -> str:
        if self._preprocessor is None:
            return query
        return self._preprocessor(query)

    def fit(self, X):
        """ Fit IDF to documents X """
        start_time = time.perf_counter()
        print(f"Fitting tf_idf vectorizer")
        X = self._preprocess_corpus(X)
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
        len_X = self.len_X
        q = self._preprocess_query(q)
        q, = super(TfidfVectorizer, self.vectorizer).transform([q])
        X = self.X_csc[:, q.indices]
        denom = X + (k1 * (1 - b + b * len_X / avdl))[:, None]
        idf = self.vectorizer._tfidf.idf_[None, q.indices] - 1.
        numer = X.multiply(np.broadcast_to(idf, X.shape)) * (k1 + 1)
        return (numer / denom).sum(1).A1
