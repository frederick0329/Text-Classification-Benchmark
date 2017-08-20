import numpy as np
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from scipy import sparse
#np.set_printoptions(threshold=np.nan)

class NBSVM:
    def __init__(self, alpha=1.0, C=1.0, beta=1.0):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.svm = LinearSVC(C=C)
        #self.svm = SVC(kernel='linear', class_weight='balanced')
        #self.svm = LogisticRegression(C=C, class_weight='balanced')
    def fit(self, X, y):
        self.r = sparse.csr_matrix(self._compute_ratios(X, y))
        X_hat = X.multiply(self.r)
        self.svm.fit(X_hat, y)
        
    def predict(self, X):
        X_hat = X.multiply(self.r)
        w = self.svm.coef_[0] 
        w_bar = np.mean(abs(w))
        self.svm.coef_[0] = (1 - self.beta) * w_bar + self.beta * w
        return self.svm.predict(X_hat)

    def _compute_ratios(self, X, y):
        p = np.sum(X[np.where( y == 1 )], axis=0) + self.alpha
        q = np.sum(X[np.where( y == -1 )], axis=0) + self.alpha
        log_p = np.log(p) - np.log(abs(p).sum())
        #p_norm = p / np.linalg.norm(p.T, 1)
        log_q = np.log(q) - np.log(abs(q).sum())
        #q_norm = q / np.linalg.norm(q.T, 1)
        r = log_p - log_q
        #print(r)
        return r


if __name__ == "__main__":
    nbsvm = NBSVM()
    X = np.random.randint(10, size=(30, 20))
    y = np.random.randint(2, size=30)
    nbsvm.fit(X, y)
    
