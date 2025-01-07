import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

# incarcarea datelor
train_path = "train.csv"
test_path = "test.csv"
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# separarea datelor de antrenament si validare
X_train, X_val, y_train, y_val = train_test_split(train_df['Text'], train_df['Label'], test_size=0.2, random_state=42)

# definirea stopwords în limba franceză
french_stop_words = [
    "le", "la", "les", "un", "une", "des", "du", "de", "et", "à", "en", "au", "aux",
    "pour", "dans", "par", "sur", "avec", "qui", "que", "quoi", "dont", "où", "mais",
    "ou", "donc", "car", "ni", "ne", "pas", "ce", "cette", "ces", "son", "sa", "ses",
    "leurs", "leur", "nos", "notre", "votre", "vos", "mon", "ma", "mes", "ton", "ta",
    "tes", "il", "elle", "ils", "elles", "nous", "vous", "je", "tu", "on", "me", "te",
    "se", "moi", "toi", "lui", "eux", "y", "en", "aussi", "bien", "comme", "être",
    "avoir", "faire", "aller", "plus", "moins", "très", "tout", "tous", "toutes",
    "quel", "quelle", "quels", "quelles", "cet", "cette", "ceux", "celles", "ça"
]

# crearea pipeline-ului cu tf-idf și random forest
pipeline = make_pipeline(
    TfidfVectorizer(stop_words=french_stop_words, max_features=5000),
    RandomForestClassifier(random_state=42, n_estimators=100)
)

# antrenarea modelului
pipeline.fit(X_train, y_train)

# evaluarea modelului pe setul de validare
y_val_pred = pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f"Acuratețea pe setul de validare: {accuracy:.2f}")
print("Raport de clasificare:")
print(classification_report(y_val, y_val_pred))

# salvarea modelului
model_path = "fake_news_model.pkl"
joblib.dump(pipeline, model_path)

# aplicarea modelului pe setul de testare
test_df['Label'] = pipeline.predict(test_df['Text'])

# salvarea setului de test completat
completed_test_path = "test_completed_random_forest.csv"
test_df.to_csv(completed_test_path, index=False)
print(f"Fișierul completat a fost salvat în: {completed_test_path}")
