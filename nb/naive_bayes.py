from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_selection import VarianceThreshold
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('spotify_songs.csv')
unique_genres = df['playlist_genre'].unique()
unique_subgenres = df['playlist_subgenre'].unique()

print(unique_genres)
print(f"Number of unique genres: {len(unique_genres)}")


print(unique_subgenres)
print(f"Number of unique sub-genres: {len(unique_subgenres)}")

X = df[['danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 
        'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
y = df['playlist_genre']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

y_train_pred = nb_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = nb_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("________________________________________________________________________")
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print(classification_report(y_test, y_test_pred))

corr = X.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

y_train_pred = nb_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = nb_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("________________________________________________________________________")
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print(classification_report(y_test, y_test_pred))
print("________________________________________________________________________")

corr = X.corr()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nb_classifier = GaussianNB()
nb_classifier.fit(X_train_scaled, y_train)

y_train_pred = nb_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_test_pred = nb_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("________________________________________________________________________")
print(f"Training Accuracy: {train_accuracy}")
print(f"Testing Accuracy: {test_accuracy}")
print(classification_report(y_test, y_test_pred))
print("________________________________________________________________________")