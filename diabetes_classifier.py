import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv('pima-indians-diabetes.csv')

X = df.drop('Outcome', axis=1)  
y = df['Outcome']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

best_k = 94  
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
best_accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, 'knn_diabetes_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print(f'Model saved with k={best_k} and accuracy={best_accuracy * 100:.2f}%')
