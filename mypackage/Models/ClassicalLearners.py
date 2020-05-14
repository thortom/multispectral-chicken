import numpy as np
import tensorflow as tf
import mypackage
import IPython
import joblib
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

timer = mypackage.Timer()

class Learner():
    def __init__(self, X_train, Y_train, saved_mode_name):
        self.saved_mode_name  = saved_mode_name
        self.X_train          = X_train
        self.Y_train          = Y_train
        self.classifier       = None
        
    def train(self):
        raise NotImplementedError
        
    def save_model(self):
        # save the model to disk
        joblib.dump(self.classifier, self.saved_mode_name)
        
    def load_model(self):
        # load the model from disk
        self.classifier = joblib.load(self.saved_mode_name)
    
    def predict(self, X, Y=None, plot=True):
        test = mypackage.StackTransform(X, Y)
        
        timer.start()
        Y_hat_stacked = self.classifier.predict(test.X_stack())
        timer.stop()

        Y_hat = test.Unstack(Y_hat_stacked, k=1)
        
        if Y is not None:
            classification = classification_report(test.Y_stack(), Y_hat_stacked)
            print(classification)

            if plot:
                selected = np.random.choice(len(Y))
                
                plt.figure(figsize=(9, 5))
                plt.subplot(121)
                plt.title("True Classes")
                plt.imshow(np.squeeze(Y[selected]))
                plt.axis('off')
                plt.subplot(122)
                plt.title("Classification")
                img = plt.imshow(np.squeeze(Y_hat[selected]))
                mypackage.Dataset._Dataset__add_legend_to_image(Y_hat[selected], img)
                plt.axis('off');

        return Y_hat

# Does not perform as well as the default SVM and LogReg
#     -> https://datascience.stackexchange.com/questions/6676/scikit-learn-getting-sgdclassifier-to-predict-as-well-as-a-logistic-regression
class SGD(Learner):
    def __init__(self, X_train, Y_train, saved_mode_name="latest_SGD_model.sav"):
        super().__init__(X_train, Y_train, saved_mode_name)
        
    def train(self, loss="hinge", penalty="l2", max_iter=20, **kwargs):
        train = mypackage.StackTransform(self.X_train, self.Y_train)

        # With loss=‘hinge’, gives a linear SVM -> https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html#sklearn.linear_model.SGDClassifier
        self.classifier = SGDClassifier(n_jobs=-1, loss=loss, penalty=penalty, **kwargs)
        timer.start()
        self.classifier.fit(train.X_stack(), train.Y_stack().ravel())
        timer.stop()

        self.save_model()

class SVM(Learner):
    def __init__(self, X_train, Y_train, C=1, kernel='rbf', gamma=1, saved_mode_name="latest_svm_model.sav"):
        super().__init__(X_train, Y_train, saved_mode_name)
        self.C      = C
        self.kernel = kernel
        self.gamma  = gamma
        
    def train(self):
        train = mypackage.StackTransform(self.X_train, self.Y_train)

        self.classifier = svm.SVC(C=self.C, kernel=self.kernel, gamma=self.gamma, decision_function_shape='ovr', verbose=True)
        timer.start()
        self.classifier.fit(train.X_stack(), train.Y_stack().ravel())
        timer.stop()

        self.save_model()
    
class LogReg(Learner): # TODO: Reduce the max_iter=50000 -to-> 5000
    def __init__(self, X_train, Y_train, C=1e5, penalty='l2', max_iter=50000, saved_mode_name="latest_LogisticReg_model.sav"):
        super().__init__(X_train, Y_train, saved_mode_name)
        self.C        = C
        self.max_iter = max_iter
        self.penalty  = penalty
        
    def train(self):
        train = mypackage.StackTransform(self.X_train, self.Y_train)

        if self.penalty == 'l2':
            solver = 'saga'
            n_jobs = -1
        if self.penalty == 'l1':
            solver = 'liblinear'
            n_jobs = None
            if self.max_iter > 3000: # Liblinear takes too long to converge
                self.max_iter = 3000
        self.classifier = LogisticRegression(C=self.C, max_iter=self.max_iter, penalty=self.penalty, solver=solver, multi_class='ovr', n_jobs=n_jobs)
        timer.start()
        self.classifier.fit(train.X_stack(), train.Y_stack().ravel())
        timer.stop()

        self.save_model()
    
def logistic_regression(X_train, Y_train, X_test, Y_test, C=1e5, plot=True, metrics=True, max_iter=10000):
    train = mypackage.StackTransform(X_train, Y_train)

    logreg = LogisticRegression(C=C, max_iter=max_iter)

    # Create an instance of Logistic Regression Classifier and fit the data.
    logreg.fit(train.X_stack(), train.Y_stack().ravel())
    
    def score(X_stacked, Y_stacked):
        weights = np.squeeze(Y_stacked-1)
        weights[weights == 2] = weights[weights == 2]*10

        # This score uses the scikit-learn accuracy_score and seems to be good metric
        score_val = logreg.score(X_stacked, Y_stacked, sample_weight=weights)
        return score_val, weights
        
    if metrics:
        test = mypackage.StackTransform(X_test, Y_test)
        Y_stacked = test.Y_stack()
        X_stacked = test.X_stack()
        Y_hat_stacked = logreg.predict(X_stacked)
        Y_hat = test.Unstack(Y_hat_stacked, k=1)

        score_val, weights = score(X_stacked, Y_stacked)
        print(score_val)
        print(f"For all test data the weighted accuracy_score with weights={np.unique(weights)} gives the score of: {score_val:.4f}")
        ################################################
        # TODO: Studdy the average='micro' vs other... #
        print(f"F1_score = {f1_score(Y_stacked, Y_hat_stacked, average='micro'):.4f}")
        print(f"Precision_score = {precision_score(Y_stacked, Y_hat_stacked, average='micro'):.4f}")
        print(f"Recall_score = {recall_score(Y_stacked, Y_hat_stacked, average='micro'):.4f}")
    
    if plot:    
        for i in range(Y_test.shape[0]):
            plt.figure(figsize=(9, 5))
            plt.subplot(1, 2, 1)
            plt.title("True Classes")
            plt.imshow(np.squeeze(Y_test[i]))
            plt.axis('off')
            plt.subplot(1, 2, 2)
            plt.title("Logistic Regression Classification")
            plt.imshow(np.squeeze(Y_hat[i]))
            plt.axis('off');
            plt.show()
            
            # This i:i+1 is used to keep the dimensions as (n_items, n, m, k)
            test = mypackage.StackTransform(X_test[i:i+1], Y_test[i:i+1])
            score_val, weights = score(test.X_stack(), test.Y_stack()) 
            print(f"Weighted accuracy_score with weights={np.unique(weights)} gives the score of: {score_val:.4f}")
            
    return logreg