# classifier/multimodal_classifier.py
"""
Multimodal classifier: trains an XGBoost classifier using
- README text (tfidf)
- concatenated commit-message embeddings / vectorized features (tfidf or embedding)
- numeric code-context features (num files, avg lines changed, language vector, etc.)

Usage:
    from classifier.multimodal_classifier import MultiModalClassifier
    clf = MultiModalClassifier()
    clf.fit(X_readme, X_commit_texts, X_code_features, y)
    preds = clf.predict(readme, commit_texts, code_features)
"""

from typing import List, Optional, Tuple
import pickle
import os

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

MODEL_DIR = os.environ.get("SMARTNOTE_MODEL_DIR", "models")

class MultiModalClassifier:
    def __init__(self,
                 readme_tfidf_max_features=20000,
                 commit_tfidf_max_features=10000,
                 xgb_params=None):
        self.readme_vec = TfidfVectorizer(max_features=readme_tfidf_max_features, ngram_range=(1,2))
        self.commit_vec = TfidfVectorizer(max_features=commit_tfidf_max_features, ngram_range=(1,2))
        self.scaler = StandardScaler()
        self.clf = xgb.XGBClassifier(
            objective="multi:softprob",
            use_label_encoder=False,
            eval_metric="mlogloss",
            **(xgb_params or {})
        )

    def _featurize(self, readmes: List[str], commit_texts: List[str], code_features: List[List[float]]):
        # readme -> tfidf
        R = self.readme_vec.transform(readmes)
        C = self.commit_vec.transform(commit_texts)
        N = np.array(code_features, dtype=float)
        if N.size == 0:
            # fallback: zeros
            N = np.zeros((R.shape[0], 1))
        Ns = self.scaler.transform(N)
        # concatenate (dense) for xgboost
        from scipy import sparse
        X = sparse.hstack([R, C, sparse.csr_matrix(Ns)], format='csr')
        return X

    def fit(self, readmes: List[str], commit_texts: List[str], code_features: List[List[float]], labels: List[str], save_path: Optional[str]=None):
        # fit vectorizers/scaler first
        self.readme_vec.fit(readmes)
        self.commit_vec.fit(commit_texts)
        import numpy as np
        self.scaler.fit(np.array(code_features, dtype=float))
        # featurize
        X = self._featurize(readmes, commit_texts, code_features)
        y = np.array(labels)
        # train/test split for debug/tracking
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.clf.fit(X_train, y_train)
        preds = self.clf.predict(X_test)
        print("Classification report (dev/test):")
        print(classification_report(y_test, preds))
        print("Accuracy:", accuracy_score(y_test, preds))
        if save_path:
            self.save(save_path)
        else:
            self.save(os.path.join(MODEL_DIR, "multimodal_clf.pkl"))

    def predict(self, readme: List[str], commit_texts: List[str], code_features: List[List[float]]):
        X = self._featurize(readme, commit_texts, code_features)
        probs = self.clf.predict_proba(X)
        preds = self.clf.predict(X)
        return preds, probs

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump({
                "readme_vec": self.readme_vec,
                "commit_vec": self.commit_vec,
                "scaler": self.scaler,
                "clf": self.clf
            }, fh)
        print("Saved multimodal classifier to", path)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        obj = cls()
        obj.readme_vec = data["readme_vec"]
        obj.commit_vec = data["commit_vec"]
        obj.scaler = data["scaler"]
        obj.clf = data["clf"]
        return obj
