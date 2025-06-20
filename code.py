import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, precision_score, recall_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

# 1) Read meteorological data
met_df = pd.read_csv("all_data_new_V3.csv")
met_df["time"] = pd.to_datetime(met_df["time"], errors="coerce")
met_df.dropna(subset=["time"], inplace=True)
met_df["date"] = met_df["time"].dt.floor("D")

# 2) Read lake area data and engineer features
area_path = Path("SouthLhonak_Area_20230101_20231025.csv")
if area_path.exists():
    area_df = pd.read_csv(area_path, header=0)
    area_df["date"] = pd.to_datetime(area_df.iloc[:, 0], errors="coerce").dt.floor("D")
    area_df.dropna(subset=["date"], inplace=True)
    area_df["lake_area_m2"] = pd.to_numeric(
        area_df.iloc[:, 1].astype(str).str.replace(",", ".", regex=False),
        errors="coerce"
    )
    area_df["lake_area_km2"] = area_df["lake_area_m2"] / 1e6
    # Remove outliers
    area_df = area_df[~area_df["lake_area_km2"].between(1.6, 1.7)]
    area_df.sort_values("date", inplace=True)
    area_df["area_diff_24h"] = area_df["lake_area_km2"].diff().fillna(0)
    # Assume avg depth of 5m
    avg_depth_m = 5.0
    area_df["avg_depth_m"] = avg_depth_m
    area_df["volume_km3"] = area_df["lake_area_km2"] * avg_depth_m / 1e3
    area_daily = (
        area_df
        .set_index("date")[['lake_area_km2','area_diff_24h','avg_depth_m','volume_km3']]
        .resample('D').mean()
    )
else:
    print("⚠️  Lake area CSV not found – area features will be skipped.")
    area_daily = pd.DataFrame(columns=['lake_area_km2','area_diff_24h','avg_depth_m','volume_km3'])

# 3) Merge dataframes
df = met_df.merge(area_daily.reset_index(), on='date', how='left')

# 4) Fill numeric NAs with column means
num_cols = df.select_dtypes(include=[np.number]).columns
for col in num_cols:
    df[col].fillna(df[col].mean(), inplace=True)

# 5) Semi-supervised labeling: Oct 3 ±1 day as flood (1), random 0 labels
flood_date = date(2023,10,3)
window = [flood_date + timedelta(d) for d in (-1,0,1)]

df['label'] = -1
mask_pos = df['date'].dt.date.isin(window)
df.loc[mask_pos,'label'] = 1
# sample negatives
neg_pool = df[df['label']==-1].index
neg_count = min(len(neg_pool), 200)
np.random.seed(42)
neg_idx = np.random.choice(neg_pool, size=neg_count, replace=False)
df.loc[neg_idx,'label'] = 0

print(f"Number of labeled 1s: {(df['label']==1).sum()}")
print(f"Number of labeled 0s: {(df['label']==0).sum()}")

# 6) Prepare features and labels
exclude = ['time','date','label']
features = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]
X = df[features].values
y = df['label'].values
labeled_mask = y!=-1
X_lab, y_lab = X[labeled_mask], y[labeled_mask]

# 7) Raw model: Self-training over RandomForest with class_weight balanced
rf = RandomForestClassifier(n_estimators=500, max_depth=5,
                            class_weight='balanced', random_state=42, n_jobs=-1)
self_clf = SelfTrainingClassifier(rf, threshold=0.9)
self_clf.fit(X, y)
proba_lab = self_clf.predict_proba(X_lab)[:,1]

# 8) Threshold analysis and Precision-Recall curve
thresholds = [0.3,0.5,0.6,0.7]
print("\n-- Threshold Analysis --")
for t in thresholds:
    preds = (proba_lab>=t).astype(int)
    p = precision_score(y_lab, preds)
    r = recall_score(y_lab, preds)
    print(f"Threshold {t}: Precision={p:.3f}, Recall={r:.3f}")

prec, rec, _ = precision_recall_curve(y_lab, proba_lab)
plt.figure(figsize=(6,4))
plt.plot(rec, prec)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (Raw Model)')
plt.tight_layout()
plt.show()

# 9) Raw model evaluation at 0.6
pred_def = (proba_lab>=0.6).astype(int)
print("\n--- Raw Model Evaluation (Threshold=0.6) ---")
print(classification_report(y_lab, pred_def, digits=3))
cm = confusion_matrix(y_lab, pred_def)
plt.figure(figsize=(4,3))
plt.imshow(cm, cmap='Blues')
for i in range(2):
    for j in range(2):
        plt.text(j,i,cm[i,j],ha='center',va='center',color='white')
plt.xticks([0,1],['0','1'])
plt.yticks([0,1],['0','1'])
plt.title('Confusion Matrix (Raw, Threshold=0.6)')
plt.tight_layout()
plt.show()
print(f"ROC-AUC: {auc(*roc_curve(y_lab, proba_lab)[:2]):.3f}")


