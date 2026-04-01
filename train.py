import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load Data
df = pd.read_csv('data/train_essays.csv') 

# 2. Convert Text to Numbers (TF-IDF)
# This looks for word patterns that AI uses more often than humans
tfidf = TfidfVectorizer(ngram_range=(1, 3), max_features=5000)
X = tfidf.fit_transform(df['text'])
y = df['generated'] # 0 for Human, 1 for AI

# 3. Split into Training & Testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train the "Brain"
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 5. Check accuracy
predictions = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")

# 6. Save for later use
joblib.dump(model, 'models/detector_model.pkl')
joblib.dump(tfidf, 'models/vectorizer.pkl')