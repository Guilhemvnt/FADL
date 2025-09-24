import os
import datetime

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------- Helper utilities ----------------------

def has_content(x):
    return int(bool(str(x).strip()))


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        # Store as tuple so sklearn can clone safely
        self.columns = tuple(columns)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=self.columns)

# ---------------------- Preprocessing ----------------------

def preprocess_dataframe(df):
    # Fill missing string columns with empty string
    string_columns = [
        'Full Name', 'Workplace', 'Location', 'Connections', 'Photo',
        'Followers', 'About', 'Experiences', 'Educations', 'Licenses', 
        'Volunteering', 'Skills', 'Recommendations', 'Projects', 'Publications',
        'Courses', 'Honors', 'Scores', 'Languages', 'Organizations', 'Interests', 'Activities'
    ]
    for col in string_columns:
        if col in df.columns:
            df[col] = df[col].fillna('')

    # Binary / presence features
    if 'Photo' in df.columns:
        df['Has_Photo'] = df['Photo'].apply(has_content)
    else:
        df['Has_Photo'] = 0

    for src, dst in [
        ('About', 'Has_About'),
        ('Projects', 'Has_Projects'),
        ('Educations', 'Has_Education'),
        ('Experiences', 'Has_Experience'),
        ('Skills', 'Has_Skills'),
        ('Licenses', 'Has_Licenses'),
        ('Interests', 'Has_Interests'),
        ('Recommendations', 'Has_Recommendations'),
    ]:
        df[dst] = df[src].apply(has_content) if src in df.columns else 0

    # Length features (use 0 for missing)
    df['About_Length'] = df.get('About', '').apply(lambda x: len(str(x)))
    df['Skills_Length'] = df.get('Skills', '').apply(lambda x: len(str(x)))
    df['Experience_Length'] = df.get('Experiences', '').apply(lambda x: len(str(x)))
    df['Education_Length'] = df.get('Educations', '').apply(lambda x: len(str(x)))
    df['Projects_Length'] = df.get('Projects', '').apply(lambda x: len(str(x)))

    # Numeric count fields - fill missing with 0 and coerce
    count_fields = [
        'Number of Experiences', 'Number of Educations', 'Number of Licenses',
        'Number of Volunteering', 'Number of Skills', 'Number of Recommendations',
        'Number of Projects', 'Number of Publications', 'Number of Courses',
        'Number of Honors', 'Number of Scores', 'Number of Languages',
        'Number of Organizations', 'Number of Interests', 'Number of Activities',
        'Connections', 'Followers'
    ]
    for col in count_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # If any of the categorical fields are missing, add empty string columns so encoder is robust
    for col in ['Location', 'Workplace', 'Full Name']:
        if col not in df.columns:
            df[col] = ''

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


# ---------------------- Visualization ----------------------

def cunf_matrix(y_test, y_pred, name="Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {name}")
    plt.show()
    plt.close()


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = getattr(model, 'coef_')
        if importances.ndim > 1:
            importances = importances[0]
    else:
        print("Model has no feature importances or coefficients.")
        return

    feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    feat_importance.plot(kind='bar')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.show()
    plt.close()


# ---------------------- Model training & evaluation ----------------------
# ---------------------- Build pipelines for ensemble ----------------------

def build_ensemble_pipelines(features, categorical_cols, numeric_cols, random_state=42):
    # Base models
    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    lr = LogisticRegression(max_iter=10000, solver='saga', random_state=random_state)

    # Preprocessor: encode categorical, optionally scale numeric
    preprocessor = build_preprocessor(categorical_cols, numeric_cols, scale_numeric=True)

    # Pipelines for individual models
    rf_pipeline = Pipeline([
        ('df_wrapper', DataFrameWrapper(features)),
        ('preprocessor', preprocessor),
        ('model', rf)
    ])
    gb_pipeline = Pipeline([
        ('df_wrapper', DataFrameWrapper(features)),
        ('preprocessor', preprocessor),
        ('model', gb)
    ])
    lr_pipeline = Pipeline([
        ('df_wrapper', DataFrameWrapper(features)),
        ('preprocessor', preprocessor),
        ('model', lr)
    ])

    # Voting Classifier
    voting = VotingClassifier(
        estimators=[('rf', rf_pipeline), ('gb', gb_pipeline), ('lr', lr_pipeline)],
        voting='soft'
    )

    # Stacking Classifier
    # Base estimators are numeric only after preprocessing, final estimator sees numeric predictions
    stacking = Pipeline([
        ('df_wrapper', DataFrameWrapper(features)),
        ('preprocessor', preprocessor),
        ('stack', StackingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            final_estimator=lr,
            passthrough=False  # avoid sending raw features to final estimator
        ))
    ])

    return rf_pipeline, gb_pipeline, lr_pipeline, voting, stacking



def build_preprocessor(categorical_cols, numeric_cols, scale_numeric=False):
    cat_pipeline = Pipeline([
        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    if scale_numeric:
        num_pipeline = Pipeline([
            ('scale', StandardScaler())
        ])
    else:
        num_pipeline = Pipeline([
            ('pass', 'passthrough')
        ])

    preprocessor = ColumnTransformer([
        ('cat', cat_pipeline, categorical_cols),
        ('num', num_pipeline, numeric_cols)
    ])

    # Ensure DataFrame output so string column names are preserved
    preprocessor.set_output(transform="pandas")

    return preprocessor



def build_pipeline(model, categorical_cols, numeric_cols, all_features, scale_numeric=False):
    preprocessor = ColumnTransformer([
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_cols),
        ('num', StandardScaler() if scale_numeric else 'passthrough', numeric_cols)
    ])
    preprocessor.set_output(transform="pandas")  # Keep column names

    return Pipeline([
        ('df_wrapper', DataFrameWrapper(all_features)),
        ('preprocessor', preprocessor),
        ('model', model)
    ])



def ensemble_classifier(dataset_path, datasetname, save_model=False, features=None,
                        use_voting=True, use_stacking=True, random_state=42):

    df = pd.read_csv(dataset_path)
    df = preprocess_dataframe(df)

    if 'Label' not in df.columns:
        raise ValueError("The dataset must contain a 'Label' column with 0 or 1 values.")

    if features is None:
        features = get_feature_list()

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in dataframe: {missing}")

    print(f"Dataset size: {df.shape[0]} rows, {df.shape[1]} columns")

    X = df[features]
    y = df['Label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=random_state
    )

    categorical_cols = [c for c in ['Location', 'Workplace', 'Full Name'] if c in X.columns]
    numeric_cols = [c for c in X.columns if c not in categorical_cols]

    rf = RandomForestClassifier(n_estimators=200, random_state=random_state)
    gb = GradientBoostingClassifier(n_estimators=200, random_state=random_state)
    lr = LogisticRegression(max_iter=10000, solver='saga', random_state=random_state)

    rf_pipeline, gb_pipeline, lr_pipeline, voting, stacking = build_ensemble_pipelines(
        features=features,
        categorical_cols=categorical_cols,
        numeric_cols=numeric_cols,
        random_state=random_state
    )

    models = {
        # 'RandomForest': rf_pipeline,
        # 'GradientBoosting': gb_pipeline,
        # 'LogisticRegression': lr_pipeline,
        # 'VotingClassifier': voting,
        'StackingClassifier': stacking
    }

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = os.path.join('results', 'model_stats')
    ensure_dir(results_dir)
    results_file = os.path.join(results_dir, f"{timestamp}_{datasetname}_ensemble_results.txt")

    with open(results_file, 'w') as f:
        for name, model in models.items():
            print(f"\nTraining {name}...")
            f.write(f"\n{name} Results:\n")

            try:
                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='roc_auc')
                cv_mean = cv_scores.mean()
                print(f"{name} CV ROC-AUC (3-fold): {cv_mean:.4f}")
                f.write(f"{name} CV ROC-AUC (3-fold): {cv_mean:.4f}\n")
            except Exception as e:
                print(f"Could not run cross_val_score for {name}: {e}")
                f.write(f"Could not run cross_val_score for {name}: {e}\n")

            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            report = classification_report(y_test, y_pred)
            print(f"\n{name} Results:\n")
            print(report)
            f.write(report + "\n")

            try:
                # voting and stacking may not implement predict_proba in the same way; handle safely
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)
                    if y_proba.ndim == 2:
                        y_proba = y_proba[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    print(f"ROC-AUC: {auc:.4f}")
                    f.write(f"ROC-AUC: {auc:.4f}\n")
            except Exception as e:
                print(f"Could not compute ROC-AUC for {name}: {e}")
                f.write(f"Could not compute ROC-AUC for {name}: {e}\n")

            cunf_matrix(y_test, y_pred, name=name)

            try:
                # extract final estimator safely
                if isinstance(model, Pipeline):
                    final_est = model.named_steps.get('model', model.steps[-1][1])
                else:
                    final_est = model

                if hasattr(final_est, 'feature_importances_'):
                    plot_feature_importance(final_est, features)
            except Exception as e:
                print(f"Could not plot feature importances for {name}: {e}")

            if save_model:
                model_filename = os.path.join(results_dir, f"{timestamp}_{datasetname}_{name}.pkl")
                try:
                    joblib.dump(model, model_filename)
                    print(f"{name} model saved to {model_filename}")
                    f.write(f"{name} model saved to {model_filename}\n")
                except Exception as e:
                    print(f"Error saving model {name}: {e}")
                    f.write(f"Error saving model {name}: {e}\n")

    print(f"\nAll results written to {results_file}")


# ---------------------- CLI / main ----------------------

def ensemble_main(dataset_paths=None):
    if dataset_paths is None:
        dataset_paths = [input('Enter the path to the dataset: ').strip()]

    for path in dataset_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
        datasetname = os.path.splitext(os.path.basename(path))[0]
        ensemble_classifier(path, datasetname, save_model=True, features=get_feature_list())


if __name__ == '__main__':
    example_paths = [
        '/home/fasma/heriot-watt/Research_Methods/FADL/datasets/trainning_Datasets/LinkedIn people profiles datasets - Clean_label_data_NoAI.csv'
    ]
    ensemble_main(example_paths)
