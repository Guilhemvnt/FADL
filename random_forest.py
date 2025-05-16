import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import joblib

import os
import datetime


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

    for col in ['Location', 'Workplace', 'Full Name', 'Connections', 'Followers']:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    count_fields = [
        'Number of Experiences', 'Number of Educations', 'Number of Licenses',
        'Number of Volunteering', 'Number of Skills', 'Number of Recommendations',
        'Number of Projects', 'Number of Publications', 'Number of Courses',
        'Number of Honors', 'Number of Scores', 'Number of Languages',
        'Number of Organizations', 'Number of Interests', 'Number of Activities'
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

def plot_feature_importance(rf, features):
    importances = rf.feature_importances_
    feat_importance = pd.Series(importances, index=features).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    feat_importance.plot(kind='bar')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()

def main():
    path = get_dataset_path()
    datasetname = os.path.splitext(os.path.basename(path))[0]
    print(f"Dataset name: {datasetname}")

    df = pd.read_csv(path)
    df = preprocess_dataframe(df)
    
    if 'Label' not in df.columns:
        raise ValueError("The dataset must contain a 'Label' column with 0 or 1 values.")

    features = get_feature_list()
    X = df[features]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    report = classification_report(y_test, y_pred)
    print(report)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")    
    results_file_name = f"{timestamp}_{datasetname}_results.txt"

    with open(results_file_name, "w") as f:
        f.write(report)

    plot_feature_importance(rf, features)
    cunf_matrix(y_test, y_pred)

    save = input("Do you want to save the model? (y/n): ").strip().lower()
    if save == 'y':
        joblib.dump(rf, f"{results_file_name}.pkl")
        print("Model saved.")

if __name__ == "__main__":
    main()
