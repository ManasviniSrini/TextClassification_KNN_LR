from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import GridSearchCV
import time

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())    
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [w for w in tokens if not w in stop_words]
    return " ".join(filtered_tokens)

# Load the dataset by name ('dair-ai/emotion' dataset)
dataset = load_dataset('dair-ai/emotion')

# Access specific splits ('train', 'validation', 'test')
train_data = dataset['train']
validation_data = dataset['validation']
test_data = dataset['test']

# Preprocess and prepare datasets
def prepare_dataset(data):
    texts = [preprocess_text(item['text']) for item in data]
    labels = [item['label'] for item in data]
    return texts, labels

X_train_texts, y_train = prepare_dataset(train_data)
X_val_texts, y_val = prepare_dataset(validation_data)
X_test_texts, y_test = prepare_dataset(test_data)

# Vectorization - fit on training data, transform on all datasets
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train_texts)
X_val = vectorizer.transform(X_val_texts)
X_test = vectorizer.transform(X_test_texts)

# Define the range of neighbors to explore
param_grid = {'n_neighbors': [1, 3, 5, 7, 9]}

# Initialize KNN model
knn = KNeighborsClassifier()

# GridSearchCV to find the best number of neighbors
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameter
print("Best number of neighbors:", grid_search.best_params_['n_neighbors'])

# Get the best KNN model
best_knn_model = grid_search.best_estimator_

# Train the best model with the whole training data
start_time=time.time()
best_knn_model.fit(X_train, y_train)
end_time=time.time()
training_time = end_time - start_time
print("Training Time:", training_time)
# Evaluate on validation data
y_val_pred = best_knn_model.predict(X_val)
print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nValidation Classification Report:\n", classification_report(y_val, y_val_pred))

# Evaluate on test data
y_test_pred = best_knn_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTest Classification Report:\n", classification_report(y_test, y_test_pred))

# Save the results to a CSV file
with open('./KNN_result.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Dataset', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
    
    # Evaluate on validation data and save
    evaluate_metrics = classification_report(y_val, y_val_pred, output_dict=True)
    writer.writerow(['Validation',
                     evaluate_metrics['accuracy'],
                     evaluate_metrics['weighted avg']['precision'],
                     evaluate_metrics['weighted avg']['recall'],
                     evaluate_metrics['weighted avg']['f1-score']])
    
    # Evaluate on test data and save
    evaluate_metrics = classification_report(y_test, y_test_pred, output_dict=True)
    writer.writerow(['Test',
                     evaluate_metrics['accuracy'],
                     evaluate_metrics['weighted avg']['precision'],
                     evaluate_metrics['weighted avg']['recall'],
                     evaluate_metrics['weighted avg']['f1-score']])
