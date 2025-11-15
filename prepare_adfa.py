

#Importing Libraries

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score,train_test_split,GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier 
from sklearn.linear_model import LogisticRegression,RidgeClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix ,classification_report,f1_score,recall_score ,roc_curve,roc_auc_score ,accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import warnings
warnings.filterwarnings("ignore")
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM




'''
#Prepare ADFA data

base_dir = Path(r"E:\Rev-DataScience\AI-ML\semi-structured\ADFA-LD") 
records = []
splits = ["Training_Data_Master", "Validation_Data_Master", "Attack_Data_Master"]

for split in splits:
    split_dir = base_dir / split
    if not split_dir.exists():
        print(f" Missing folder: {split_dir}")
        continue

    if "attack" in split.lower():
        for root, dirs, files in os.walk(split_dir):
            for f in files:
                fp = Path(root) / f
                text = fp.read_text(errors="ignore").strip()
                records.append({
                    "split": "attack",
                    "subpath": str(fp.relative_to(base_dir)),
                    "file": f,
                    "text": text,
                    "label": 1
                })
    else:
        for root, dirs, files in os.walk(split_dir):
            for f in files:
                fp = Path(root) / f
                text = fp.read_text(errors="ignore").strip()
                records.append({
                    "subpath": str(fp.relative_to(base_dir)),
                    "file": f,
                    "text": text,
                    "label": 0
                })

print("Records collected:", len(records))
if records:
    print("Example record:", records[0])

df = pd.DataFrame(records)
print(df.groupby(["split", "label"]).size())
df.to_csv("adfa_parsed.csv", index=False)
print(" Saved to adfa_parsed.csv")
'''

#Load Data & Info

df_=pd.read_csv(r'E:\Rev-DataScience\AI-ML\semi-structured\adfa_parsed.csv',encoding='latin')

df=df_.copy()

print(df.head(5))

print(df.shape) #5951 , 5
print(df['split'].value_counts())
print(df.dtypes)
print(df.isnull().sum()) #zero
print(df.memory_usage(deep=True))

df=df.drop(['subpath','file'],axis=1)
print(df.memory_usage(deep=True).sum())
print(df.duplicated().sum()) #2019
df=df.drop_duplicates()
print(df.duplicated().sum())
print(df.columns)

#Relatins
print('Relations Between Split and Label :- \n',df.groupby('split')['label'].mean())
print('----------------------')

df['length'] = df['text'].apply(lambda x: len(str(x).split()))
df['unique_calls'] = df['text'].apply(lambda x: len(set(str(x).split())))
df['mean_call'] = df['text'].apply(lambda x: np.mean(list(map(int, str(x).split()))))
df[['split', 'label', 'length', 'unique_calls', 'mean_call']].head()


for col in df.columns:
    if df[col].dtype == 'object':
        df[col]=df[col].astype('category')
    else:
        df[col]=df[col].astype('int8')

print(df.memory_usage(deep=True).sum())

"""
Overview Insights
------------------------

-- values_count --->

Validation data = 4372
Training data   = 833
attack data     = 746

#------

-- Memory Usage (Before) = 10357652


#------

-- Empty & Duplicated Values--->
There are no null or duplicated values

#------


Actions :-
-------------------------
optimized data object -> category and int64 -> int8
kept complicated columns in object
droped not useful column [subpath - file]
-- Mempry Usage (After)  = 7048775
-- Duplicated after editing 2019 and removed 

"""

#-----------------------------------------
# Chhosing Model
#-----------------------------------------

vec = TfidfVectorizer(token_pattern=r"\d+")
x = vec.fit_transform(df['text'])
y=df['label'].values
x_train, x_test,y_train , y_test=train_test_split(x,y,test_size=0.2,random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=500),
    "LinearSVM": SVC(kernel="linear"),
    "RandomForest": RandomForestClassifier(),
    "XGBoost": XGBClassifier(),
    "NaiveBayes": MultinomialNB(),
}


baseline=[]

for name,model in models.items():
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    acc=accuracy_score(y_test,y_pred)
    f1=f1_score(y_test,y_pred)
    recall=recall_score(y_test,y_pred)
    cs=confusion_matrix(y_test,y_pred)

    baseline.append({'Model':name,'ConfMatrx':cs,'F1':f1,'Recall':recall})

basl=pd.DataFrame(baseline).sort_values(by='F1',ascending=False)

#Best Score Model in scoring => Random Forest F1= 0.84

#Applying & Tuning Model
model=RandomForestClassifier(random_state=42)

ns=[100,200,300,400]
for n in ns:
    model=RandomForestClassifier(n_estimators=n,random_state=42)
    model.fit(x_train,y_train)
    y_pred=model.predict(x_test)
    f1=f1_score(y_test,y_pred)
    print(f'when n = {n} ,F1 = {f1}')

#when n_estimator = 100
parms={
    'max_depth':[2,3,4,5,6],
    'min_samples_split':[2,5,7]
}
grid=GridSearchCV(model,param_grid=parms,cv=5,n_jobs=-1,verbose=1)
grid.fit(x_train,y_train)

print(grid.best_params_)

#when max_depth =6 , min samples split =5

model=RandomForestClassifier(n_estimators=100,max_depth=6,min_samples_split=5,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
y_prob=model.predict_proba(x_test)[:,1]

scores=cross_val_score(model,x,y,cv=5,scoring='f1')
print('Average F1',round(scores.mean(),2))


# ------------------------------- 
# Visualization Of Random Forest
# ------------------------------- 

cs=confusion_matrix(y_test,y_pred)
plt.figure(figsize=(9,6))
sns.heatmap(cs,fmt='d',annot=True,cmap='coolwarm') 
plt.title('Confusion Matrix (Random Forest Model)')
plt.savefig("confusion_matrix_RandomForest.png", dpi=150)
plt.show() 
#--------
auc=roc_auc_score(y_test,y_prob)
plt.figure(figsize=(8,6)) 
fpr,tpr,_=roc_curve(y_test,y_prob)
plt.plot(fpr,tpr,label=f'Roc Auc Score : {auc:.2f}') 
plt.plot([0,1],[0,1],'k--') 
plt.xlabel('False Positive Rate') 
plt.ylabel('True Positive Rate') 
plt.title('Visualization Of Roc Curve (Random Forest Model)') 
plt.legend()
plt.savefig("Roc_Curve_RandomForest.png", dpi=150)
plt.show() 
#----------
plt.figure(figsize=(8,5))
sns.countplot(x='label',data=df) 
plt.title('Distribution of Label (Random Forest Model)') 
plt.savefig("distribution_RandomForest.png", dpi=150)
plt.show()


# ---------------------------------------
# Update and Fixing Model For ADFA data
# ---------------------------------------
CSV_PATH = Path("adfa_parsed.csv")
df = pd.read_csv(CSV_PATH, encoding="latin1")
if {'subpath','file'}.issubset(df.columns):
    df = df.drop(columns=['subpath','file'])
df = df.drop_duplicates(subset=['text','label']).reset_index(drop=True)

# -------------------------
# Feature Engineering
# -------------------------
df['length'] = df['text'].apply(lambda x: len(str(x).split()))
df['unique_calls'] = df['text'].apply(lambda x: len(set(str(x).split())))
df['mean_call'] = df['text'].apply(lambda x: np.mean([int(t) for t in str(x).split() if t.isdigit()]))
numeric_features = ['length','unique_calls','mean_call']

# -------------------------
# Realistic Augmentation
# -------------------------
# Normal: sequences
normal_df = df[df['label']==0].copy()
# ================= NORMAL AUG ======================
# Normal: sequences  1–500

normal_df = df[df['label'] == 0].copy()

normal_aug = normal_df.sample(n=80, replace=True)

def add_realistic_normal_noise(text):
    extra = np.random.randint(1, 500, size=np.random.randint(3, 10))
    return text + ' ' + ' '.join(map(str, extra))

normal_aug['text'] = normal_aug['text'].apply(add_realistic_normal_noise)


df = pd.concat([df, normal_aug], ignore_index=True)
# ================= ANOMALY AUG ======================
# Anomaly: sequences 
anomaly_texts = [

    # 1)  out-of-range  → anomaly 100%
    ' '.join(map(str, np.random.randint(20000, 5000000, size=60))),

    # 2)  syscall IDs
    ' '.join(map(str, np.random.randint(-800, -1, size=40))),

    # 3) digits–5 digits = suspicious but realistic anomaly
    ' '.join(map(str, np.random.randint(1000, 99999, size=50))),

    ' '.join(map(str, np.random.randint(1, 500, size=130))),

    ' '.join(
        list(map(str, np.random.randint(1, 500, size=20))) +
        [str(np.random.randint(2000000, 8000000))] +
        list(map(str, np.random.randint(1, 500, size=20)))
    ),

    # 6) ultra large numbers (extreme anomaly)
    ' '.join(map(str, np.random.randint(3000000, 50000000, size=70)))
]

anomaly_aug = pd.DataFrame({
    'text': anomaly_texts,
    'label': [1] * len(anomaly_texts)
})


# ---------------------------------------------------
#  Extra Targeted Augmentation (Fix Model Mistakes)
# ---------------------------------------------------

# Normal short sequences (to stop marking short sequences as anomaly)
normal_short = pd.DataFrame({
    'text': [
        "6 11 45 33 192 33 7 9",
        "12 45 7",
        "1 2 3 4 5 6 7 8",
        "10 20 30 40",
        "3 5 7 9 11",
        "50 60 70"
    ],
    'label': [0, 0, 0, 0, 0, 0]
})

# Specific anomalies that model got wrong
anomaly_specific = pd.DataFrame({
    'text': [
        "99999 454545454 9999 777 12 54 45 99 99 9",
        "889991 77777 44 55 66"
    ],
    'label': [1, 1]
})

# Add them to df
df = pd.concat([df, normal_short, anomaly_specific], ignore_index=True)

# -------------------------
#  Train/Test split
# -------------------------
X = df[['text'] + numeric_features]
y = df['label'].values


# -----------------------------
#  ColumnTransformer + XGBoost
# -----------------------------
hashing = HashingVectorizer(token_pattern=r"\d+", n_features=2**16)
numeric_transformer = Pipeline([("scaler", StandardScaler())])
preprocessor = ColumnTransformer([
    ("text_hash", hashing, "text"),
    ("numeric", numeric_transformer, numeric_features)
])

#Applaying Best Model

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42
)

pipe = Pipeline([
    ("preproc", preprocessor),
    ("clf", xgb)
])



# -------------------------
#  Train
# ------------------------- 
pipe.fit(X, y)
print("XGBoost model trained on augmented ADFA data.")

# =============================
#   NEW VISUALIZATION FOR XGB
# =============================

# Prepare test set for final model visualization
X_full = df[['text'] + numeric_features]
y_full = df['label'].values

x_train_f, x_test_f, y_train_f, y_test_f = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42
)

# Predict using final pipeline
y_pred_f = pipe.predict(x_test_f)
y_prob_f = pipe.predict_proba(x_test_f)[:, 1]

# -------------------------
# Confusion Matrix
# -------------------------
plt.figure(figsize=(9,6))
cm = confusion_matrix(y_test_f, y_pred_f)
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm")
plt.title("Confusion Matrix - XGBoost Final Model")
plt.savefig("confusion_matrix_XGBoost.png", dpi=150)
plt.show()

# -------------------------
# ROC Curve
# -------------------------
fpr, tpr, _ = roc_curve(y_test_f, y_prob_f)
auc = roc_auc_score(y_test_f, y_prob_f)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
plt.plot([0,1], [0,1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - XGBoost Final Model")
plt.legend()
plt.savefig("Roc_Curve_XGBoost.png", dpi=150)
plt.show()

# -------------------------
# Label Distribution
# -------------------------

plt.figure(figsize=(7,5))
sns.countplot(x=df['label'])
plt.title("Label Distribution After Augmentation")
plt.savefig("distribution_XGBoost.png", dpi=150)
plt.show()



# -------------------------
#  Save pipeline
# -------------------------
OUT_DIR = Path("E:\Rev-DataScience\AI-ML\semi-structured")
OUT_DIR.mkdir(exist_ok=True, parents=True)
joblib.dump(pipe, OUT_DIR / "adfa_xgb_pipeline.pkl")
print("Saved pipeline to:", OUT_DIR / "adfa_xgb_pipeline.pkl")
