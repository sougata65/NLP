|Model|Vectorizer|Accuracy|F1(Macro)|Hyperparameters|
|---|---|---|---|---|
|MultinomialNB|CountVectorizer|0.821|0.820999820999821|None|
|MultinomialNB|TfidfVectorizer|0.818|0.8179883512544802|None|
|GaussianNB|CountVectorizer|0.692|0.6917225502952657|None|
|GaussianNB|TfidfVectorizer|0.695|0.694706613055146|None|
|LinearSVC|CountVectorizer|0.871|0.8709967749193729|{'C': 0.01}|
|LinearSVC|TfidfVectorizer|0.869|0.8689893881404394|{'C': 0.1}|
|LinearSVC|TfidfVectorizer|0.865|0.8649836630232257|{'C': 1}|
|SVC|TfidfVectorizer| - | 0.873|{'C':1, kernel: Linear}
|SVC|Tfidf|0.876|0.875987598759876|{'C': 1, 'kernel': 'rbf', 'gamma': 1}|
|SVC|Tfidf|0.877|0.8769851151989391|{'C': 10, 'kernel': 'rbf', 'gamma': 1}|
|SVC|Tfidf|0.871|0.8709781953150082|{'C': 100, 'kernel': 'rbf', 'gamma': 0.01}|
|SVC|Tfidf|0.877|0.8769851151989391|{'C': 100, 'kernel': 'rbf', 'gamma': 1}|
