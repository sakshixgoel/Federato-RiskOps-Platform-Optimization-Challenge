import pandas as pd
import ast
import numpy as np
from datetime import timedelta

file_path = '2024combined_file.csv'
df = pd.read_csv(file_path)

df['server_received_time'] = pd.to_datetime(df['server_received_time'])
df = df.sort_values(by=['user_id', 'server_received_time'])

df['time_since_last'] = df.groupby('user_id')['server_received_time'].diff()
df['is_new_session'] = df['time_since_last'].isna() | (df['time_since_last'] > pd.Timedelta(minutes=30))
df['session_id'] = df.groupby('user_id')['is_new_session'].cumsum().astype(str) + "_" + df['user_id']
df = df.sort_values(by=['session_id', 'server_received_time'])

df['next_session_time'] = df.groupby('user_id')['server_received_time'].shift(-1)
df['returned'] = (df['next_session_time'] - df['server_received_time']) > pd.Timedelta(days=1)
retention_rates = df.groupby('event_type')['returned'].mean().reset_index()
retention_rates.columns = ['event_type', 'retention_probability']
df = df.merge(retention_rates, on='event_type', how='left')

df['next_action'] = df.groupby('session_id')['event_type'].shift(-1)
df = df.merge(retention_rates, left_on='next_action', right_on='event_type', how='left', suffixes=('', '_next'))
df['y_best'] = df.groupby('event_type')['event_type_next'].transform(
    lambda x: x.mode()[0] if not x.isna().all() else "session_end"
)
df = df[['session_id', 'user_id', 'server_received_time', 'event_type', 'event_properties', 'y_best']]
df['y_best'] = df['y_best'].fillna("session_end")
df = df.sort_values(by=['session_id', 'server_received_time'])

def extract_property(event_json, key):
    try:
        parsed = ast.literal_eval(event_json)
        return parsed.get(key, None)
    except (ValueError, SyntaxError):
        return None

df['event_slug'] = df['event_properties'].apply(lambda x: extract_property(x, 'slug'))
df['event_display_name'] = df['event_properties'].apply(lambda x: extract_property(x, 'displayName'))
df['y_best_detail'] = df.groupby('session_id')['event_slug'].shift(-1)
df['y_best_display'] = df.groupby('session_id')['event_display_name'].shift(-1)
df['y_best'] = df['y_best'].astype(str) + "::" + df['y_best_detail'].astype(str)
df['y_best'] = df['y_best'].fillna(df['event_type'])

generic_actions = ["application-window-opened", "session_start", "session_end", "::None"]
df = df[~df['y_best'].isin(generic_actions)]
df = df[~df['y_best'].str.endswith("::None")]
df = df[~df['y_best'].str.endswith("widget:render")]
df = df[~df['y_best'].str.endswith("widget:render::None")]
df.loc[df['event_type'] == df['y_best'], 'y_best'] = "session_end"
df['y_best'] = df['y_best'].str.replace("::nan", "", regex=False)

df[['primary_y', 'detail_y']] = df['y_best'].str.split("::", n=1, expand=True)
df['detail_y'] = df['detail_y'].fillna("None")

df = df[df['detail_y'] != "None"]

df['session_length'] = df.groupby('session_id')['server_received_time'].transform(
    lambda x: (x.max() - x.min()).total_seconds()
)
df['time_since_last_session'] = df.groupby('user_id')['server_received_time'].diff().dt.total_seconds()
df['total_past_sessions'] = df.groupby('user_id')['session_id'].transform('nunique')

df['event_category'] = df['event_type'].apply(lambda x: x.split(":")[1] if ":" in x else x)
df['hour_of_day'] = df['server_received_time'].dt.hour
df['day_of_week'] = df['server_received_time'].dt.dayofweek  
df['is_working_hours'] = df['hour_of_day'].apply(lambda x: 1 if 9 <= x <= 18 else 0)

df['server_received_time_numeric'] = df['server_received_time'].astype(np.int64) // 10**9

df['previous_event_type'] = df.groupby('session_id')['event_type'].shift(1)
df['time_since_previous_event'] = df.groupby('session_id')['server_received_time_numeric'].diff()
df['session_event_count'] = df.groupby('session_id')['event_type'].transform('count')

df['previous_event_type'].fillna("None", inplace=True)
df['time_since_previous_event'].fillna(0, inplace=True)
df['session_event_count'].fillna(1, inplace=True)

df = df[['session_id'] + list(df.columns.difference(['session_id']))] 

start_date = pd.to_datetime(df['server_received_time_numeric'].min(), unit='s')
cutoff_date = start_date + pd.Timedelta(days=28)
cutoff_numeric = int(cutoff_date.timestamp())
df_first_28 = df[df['server_received_time_numeric'] < cutoff_numeric]

features = [
    'session_length', 'time_since_last_session', 'total_past_sessions',
    'event_type', 'event_category', 'event_slug', 'event_display_name',
    'hour_of_day', 'day_of_week', 'is_working_hours', 'server_received_time_numeric',
    'previous_event_type', 'time_since_previous_event', 'session_event_count'  
]
target = 'y_best'
df = df[features + [target]]

df['time_since_last_session'].fillna(0, inplace=True)
df.loc[df['session_length'] == 0, 'session_length'] = 1
df['total_past_sessions'] = df['total_past_sessions'].fillna(0).astype(int)

print("Total records AFTER Removing 'None':", len(df))
print("Total records in First 28 Days:", len(df_first_28))

X_prepared = df_first_28[features]
y_prepared = df_first_28[target]


df_first_28 = df_first_28[['session_id'] + list(df_first_28.columns.difference(['session_id']))]

df_first_28[['primary_y', 'detail_y']] = df_first_28['y_best'].str.split("::", n=1, expand=True)
df_first_28['detail_y'] = df_first_28['detail_y'].fillna("None")

df_first_28 = df_first_28[df_first_28['detail_y'] != "None"]

if 'session_id' not in df_first_28.columns:
    raise KeyError("Error: session_id is missing before computing contextual features!")

print("Checking if session_id exists:", 'session_id' in df_first_28.columns)

df_first_28['previous_event_type'] = df_first_28.groupby('session_id')['event_type'].shift(1)
df_first_28['time_since_previous_event'] = df_first_28.groupby('session_id')['server_received_time'].diff()
df_first_28['session_event_count'] = df_first_28.groupby('session_id')['event_type'].transform('count')

df_first_28['previous_event_type'].fillna("None", inplace=True)
df_first_28['time_since_previous_event'].fillna(0, inplace=True)
df_first_28['session_event_count'].fillna(1, inplace=True)

print("Columns in df_first_28:", df_first_28.columns)


features = [
    'session_length', 'time_since_last_session', 'total_past_sessions',
    'event_type', 'event_category', 'event_slug', 'event_display_name',
    'hour_of_day', 'day_of_week', 'is_working_hours', 'server_received_time_numeric'
]

X = df_first_28[features]
y_primary = df_first_28['primary_y']
y_detail = df_first_28['detail_y']

from sklearn.model_selection import train_test_split

X_train, X_val, y_primary_train, y_primary_val = train_test_split(X, y_primary, test_size=0.2, random_state=42)
y_detail_train = y_detail.loc[X_train.index]  
y_detail_val = y_detail.loc[X_val.index]

print("X_train shape:", X_train.shape)
print("y_primary_train shape:", y_primary_train.shape)
print("y_detail_train shape:", y_detail_train.shape)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_features = [
    'session_length', 'time_since_last_session', 'total_past_sessions',
    'hour_of_day', 'day_of_week', 'is_working_hours', 'server_received_time_numeric'
]
categorical_features = ['event_type', 'event_category', 'event_slug', 'event_display_name']

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

X_train_transformed = preprocessor.fit_transform(X_train)

X_val_transformed = preprocessor.transform(X_val)

print("X_train_transformed shape:", X_train_transformed.shape)
print("X_val_transformed shape:", X_val_transformed.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler 
from sklearn.metrics import accuracy_score, classification_report

undersampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
X_train_balanced, y_detail_train_balanced = undersampler.fit_resample(X_train_transformed, y_detail_train)


primary_pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

detail_pipeline = Pipeline(steps=[
    ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
])

primary_pipeline.fit(X_train_transformed, y_primary_train)
detail_pipeline.fit(X_train_balanced, y_detail_train_balanced)

y_primary_pred = primary_pipeline.predict(X_val_transformed)
y_detail_pred = detail_pipeline.predict(X_val_transformed)

print("Primary Model Training Accuracy:", primary_pipeline.score(X_train_transformed, y_primary_train))
print("Primary Model Validation Accuracy:", accuracy_score(y_primary_val, y_primary_pred))
print("\nPrimary Model Classification Report:")
print(classification_report(y_primary_val, y_primary_pred))

print("Detail Model Training Accuracy:", detail_pipeline.score(X_train_balanced, y_detail_train_balanced))
print("Detail Model Validation Accuracy:", accuracy_score(y_detail_val, y_detail_pred))
print("\nDetail Model Classification Report:")
print(classification_report(y_detail_val, y_detail_pred))
