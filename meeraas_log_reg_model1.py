import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = 'diabetic_data.csv'  
data = pd.read_csv(file_path)

# Selecting features (X) and target variable (y)
features = ['diag_1', 'diag_2', 'diag_3']
target = 'readmitted'

X = data[features]
y = data[target]

# converting categorical variables to numeric 
label_encoder = LabelEncoder()
X['diag_1'] = label_encoder.fit_transform(X['diag_1'])
X['diag_2'] = label_encoder.fit_transform(X['diag_2'])
X['diag_3'] = label_encoder.fit_transform(X['diag_3'])
y = label_encoder.fit_transform(y)

# spliting  data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')
