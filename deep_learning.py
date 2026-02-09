import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# === Provided Utility Functions ===

def get_dataset_path():
    path = input("Enter the path to the dataset: ").strip()
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    return path

def has_content(x): 
    return int(bool(str(x).strip()))

def preprocess_dataframe(df):
    string_columns = [
        'Full Name', 'Workplace', 'Location', 'Connections', 'Photo',
        'Followers', 'About', 'Experiences', 'Educations', 'Licenses', 
        'Volunteering', 'Skills', 'Recommendations', 'Projects', 'Publications',
        'Courses', 'Honors', 'Scores', 'Languages', 'Organizations', 'Interests', 'Activities'
    ]
    df[string_columns] = df[string_columns].fillna('')
    
    df['Has_Photo'] = df['Photo']
    df['Has_About'] = df['About'].apply(has_content)
    df['Has_Projects'] = df['Projects'].apply(has_content)
    df['Has_Education'] = df['Educations'].apply(has_content)
    df['Has_Experience'] = df['Experiences'].apply(has_content)
    df['Has_Skills'] = df['Skills'].apply(has_content)
    df['Has_Licenses'] = df['Licenses'].apply(has_content)
    df['Has_Interests'] = df['Interests'].apply(has_content)
    df['Has_Recommendations'] = df['Recommendations'].apply(has_content)

    df['About_Length'] = df['About'].apply(len)
    df['Skills_Length'] = df['Skills'].apply(len)
    df['Experience_Length'] = df['Experiences'].apply(len)
    df['Education_Length'] = df['Educations'].apply(len)
    df['Projects_Length'] = df['Projects'].apply(len)

    for col in ['Location', 'Workplace', 'Full Name']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    count_fields = [
        'Number of Experiences', 'Number of Educations', 'Number of Licenses',
        'Number of Volunteering', 'Number of Skills', 'Number of Recommendations',
        'Number of Projects', 'Number of Publications', 'Number of Courses',
        'Number of Honors', 'Number of Scores', 'Number of Languages',
        'Number of Organizations', 'Number of Interests', 'Number of Activities',
        'Connections', 'Followers'
    ]   
    df[count_fields] = df[count_fields].fillna(0)
    
    return df

def get_feature_list():
    return [
        'Has_Photo', 'Has_About', 'Has_Projects', 'Has_Education', 'Has_Experience',
        'Has_Skills', 'Has_Licenses', 'Has_Interests', 'Has_Recommendations',
        'About_Length', 'Skills_Length', 'Experience_Length', 'Education_Length', 'Projects_Length',
        'Location', 'Workplace', 'Full Name', 'Connections', 'Followers',
        'Number of Experiences', 'Number of Educations', 'Number of Licenses',
        'Number of Volunteering', 'Number of Skills', 'Number of Recommendations',
        'Number of Projects', 'Number of Publications', 'Number of Courses',
        'Number of Honors', 'Number of Scores', 'Number of Languages',
        'Number of Organizations', 'Number of Interests', 'Number of Activities'
    ]

def cunf_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# === Deep Learning Pipeline ===

def train_deep_learning_model():
    #path = get_dataset_path()
    path = "/home/fasma/heriot-watt/Research_Methods/FADL/datasets/trainning_Datasets/LinkedIn people profiles datasets - Clean_Original_label_data.csv"
    df = pd.read_csv(path)

    # Assumes 'Label' column exists: 0 = real, 1 = fake
    if 'Label' not in df.columns:
        raise ValueError("Dataset must contain a 'Label' column.")

    df = preprocess_dataframe(df)
    features = get_feature_list()
    
    X = df[features]
    y = df['Label']

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build the model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, validation_split=0.2, epochs=2000, batch_size=32, verbose=1)

    # Evaluate
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.2f}")

    # Predictions and Confusion Matrix
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    cunf_matrix(y_test, y_pred)

    return model, history

# === Run the training ===
if __name__ == "__main__":
    #check time execution
    
    start_time = time.time()
    
    train_deep_learning_model()

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")