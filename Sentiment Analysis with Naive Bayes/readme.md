|Model|Vectorizer|Accuracy|F1(Macro)|Hyperparameters|
|---|---|---|---|---|
|MultinomialNB|CountVectorizer|0.821|0.820999820999821|None|
|MultinomialNB|TfidfVectorizer|0.818|0.8179883512544802|None|
|GaussianNB|CountVectorizer|0.692|0.6917225502952657|None|
|GaussianNB|TfidfVectorizer|0.695|0.694706613055146|None|
|LinearSVC|CountVectorizer|0.852|0.8519976319621114|{'C': 0.001}|
|LinearSVC|CountVectorizer|0.871|0.8709967749193729|{'C': 0.01}|
|LinearSVC|CountVectorizer|0.86|0.8599859985998599|{'C': 0.1}|
|LinearSVC|TfidfVectorizer|0.833|0.8329797905546571|{'C': 0.01}|
|LinearSVC|TfidfVectorizer|0.869|0.8689893881404394|{'C': 0.1}|
|LinearSVC|TfidfVectorizer|0.865|0.8649836630232257|{'C': 1}|
|SVC|TfidfVectorizer| - | 0.873|{'C':1, kernel: Linear}
|SVC|Tfidf|0.765|0.7593670226328115|{'C': 0.1, 'kernel': 'rbf', 'gamma': 0.1}|
|SVC|Tfidf|0.829|0.828979306496086|{'C': 0.1, 'kernel': 'rbf', 'gamma': 1}|
|SVC|Tfidf|0.798|0.7974523110490768|{'C': 1, 'kernel': 'rbf', 'gamma': 0.01}|
|SVC|Tfidf|0.861|0.8609931886662446|{'C': 1, 'kernel': 'rbf', 'gamma': 0.1}|
|SVC|Tfidf|0.876|0.875987598759876|{'C': 1, 'kernel': 'rbf', 'gamma': 1}|
|SVC|Tfidf|0.799|0.7984758356485218|{'C': 10, 'kernel': 'rbf', 'gamma': 0.001}|
|SVC|Tfidf|0.864|0.8639804131794978|{'C': 10, 'kernel': 'rbf', 'gamma': 0.01}|
|SVC|Tfidf|0.873|0.8729785333721398|{'C': 10, 'kernel': 'rbf', 'gamma': 0.1}|
|SVC|Tfidf|0.877|0.8769851151989391|{'C': 10, 'kernel': 'rbf', 'gamma': 1}|
|SVC|Tfidf|0.799|0.7984758356485218|{'C': 100, 'kernel': 'rbf', 'gamma': 0.0001}|
|SVC|Tfidf|0.864|0.8639804131794978|{'C': 100, 'kernel': 'rbf', 'gamma': 0.001}|
|SVC|Tfidf|0.871|0.8709781953150082|{'C': 100, 'kernel': 'rbf', 'gamma': 0.01}|
|SVC|Tfidf|0.849|0.8489817267889415|{'C': 100, 'kernel': 'rbf', 'gamma': 0.1}|
|SVC|Tfidf|0.877|0.8769851151989391|{'C': 100, 'kernel': 'rbf', 'gamma': 1}|
