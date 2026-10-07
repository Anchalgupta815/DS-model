"""Breast Cancer Diagnosis Prediction - full pipeline (EDA, preparation, modelling, saving)."""
import json, warnings
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid"); SEED = 42
R = {}

# ---------- 1. Load ----------
raw = load_breast_cancer(as_frame=True)
df = raw.frame.copy()
df["diagnosis"] = (df["target"] == 0).astype(int)      # 1 = Malignant, 0 = Benign
df = df.drop(columns="target")
feat = [c for c in df.columns if c != "diagnosis"]
R["shape"] = df.shape
R["class_counts"] = df["diagnosis"].value_counts().to_dict()

# ---------- 2. Data preparation checks ----------
R["missing_total"] = int(df.isna().sum().sum())
R["duplicates"] = int(df.duplicated().sum())
out = {}
for c in feat:
    q1, q3 = df[c].quantile([.25, .75]); i = q3 - q1
    out[c] = int(((df[c] < q1 - 1.5*i) | (df[c] > q3 + 1.5*i)).sum())
out = pd.Series(out).sort_values(ascending=False)
R["outliers_top"] = out.head(6).to_dict()
R["outlier_rows_any"] = int(((df[feat] < (df[feat].quantile(.25) - 1.5*(df[feat].quantile(.75)-df[feat].quantile(.25)))) |
                             (df[feat] > (df[feat].quantile(.75) + 1.5*(df[feat].quantile(.75)-df[feat].quantile(.25))))).any(axis=1).sum())

# ---------- 3. EDA ----------
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
cc = df["diagnosis"].map({0: "Benign", 1: "Malignant"}).value_counts()
ax[0].bar(cc.index, cc.values, color=["#4C9F70", "#D1495B"]); ax[0].set_title("Class distribution")
for i, v in enumerate(cc.values): ax[0].text(i, v+4, f"{v} ({v/len(df):.1%})", ha="center")
ax[1].pie(cc.values, labels=cc.index, colors=["#4C9F70", "#D1495B"], autopct="%1.1f%%"); ax[1].set_title("Share")
plt.tight_layout(); plt.savefig("figures/01_class_distribution.png", dpi=150); plt.close()

key = ["mean radius", "mean perimeter", "mean concavity", "mean concave points"]
fig, ax = plt.subplots(1, 4, figsize=(13, 3.4))
for a, c in zip(ax, key):
    sns.histplot(data=df, x=c, hue=df["diagnosis"].map({0: "Benign", 1: "Malignant"}), kde=True, ax=a,
                 palette=["#4C9F70", "#D1495B"], legend=(c == key[-1]))
    a.set_title(c)
plt.tight_layout(); plt.savefig("figures/02_distributions.png", dpi=150); plt.close()

fig, ax = plt.subplots(1, 4, figsize=(12, 3.4))
for a, c in zip(ax, key):
    sns.boxplot(data=df, x=df["diagnosis"].map({0: "Benign", 1: "Malignant"}), y=c, ax=a,
                palette=["#4C9F70", "#D1495B"]); a.set_xlabel(""); a.set_title(c)
plt.tight_layout(); plt.savefig("figures/03_boxplots.png", dpi=150); plt.close()

mean_cols = [c for c in feat if c.startswith("mean")]
plt.figure(figsize=(8, 6.5))
sns.heatmap(df[mean_cols + ["diagnosis"]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 6})
plt.title("Correlation heatmap (mean features + diagnosis)"); plt.tight_layout()
plt.savefig("figures/04_correlation_heatmap.png", dpi=150); plt.close()

corr_t = df[feat].corrwith(df["diagnosis"]).sort_values()
R["top_target_corr"] = corr_t.abs().sort_values(ascending=False).head(6).round(3).to_dict()
top = corr_t.reindex(corr_t.abs().sort_values(ascending=False).head(10).index).sort_values()
plt.figure(figsize=(7, 4)); plt.barh(top.index, top.values, color=["#D1495B" if v > 0 else "#4C9F70" for v in top.values])
plt.title("Top 10 features by correlation with Malignant"); plt.tight_layout()
plt.savefig("figures/05_target_correlation.png", dpi=150); plt.close()

# multicollinearity
cm = df[feat].corr().abs(); pairs = (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool)).stack())
R["pairs_gt_095"] = int((pairs > 0.95).sum())
drop = [c for c in cm.columns if (cm.where(np.triu(np.ones(cm.shape), 1).astype(bool))[c] > 0.95).any()]
R["dropped_features"] = drop
sel = [c for c in feat if c not in drop]
R["n_selected"] = len(sel)

# ---------- 4. Modelling ----------
X, y = df[feat], df["diagnosis"]
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
R["train_test"] = [len(Xtr), len(Xte)]
models = {
    "Logistic Regression": LogisticRegression(max_iter=5000),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "SVM (RBF)": SVC(probability=True, random_state=SEED),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    "Gradient Boosting": GradientBoostingClassifier(random_state=SEED),
}
cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows = []
for name, m in models.items():
    pipe = Pipeline([("sc", StandardScaler()), ("m", m)])
    s = cross_validate(pipe, Xtr, ytr, cv=cv, scoring=["accuracy", "recall", "roc_auc"])
    pipe.fit(Xtr, ytr); p = pipe.predict(Xte); pr = pipe.predict_proba(Xte)[:, 1]
    rows.append(dict(Model=name, CV_Accuracy=s["test_accuracy"].mean(), CV_Recall=s["test_recall"].mean(),
                     CV_AUC=s["test_roc_auc"].mean(), Test_Accuracy=accuracy_score(yte, p),
                     Precision=precision_score(yte, p), Recall=recall_score(yte, p),
                     F1=f1_score(yte, p), ROC_AUC=roc_auc_score(yte, pr)))
res = pd.DataFrame(rows).round(4); res.to_csv("model_comparison.csv", index=False)
print(res.to_string())

# Tune the best two candidates
grids = {
    "SVM (RBF)": (SVC(probability=True, random_state=SEED), {"m__C": [0.1, 1, 10, 100], "m__gamma": ["scale", 0.01, 0.001]}),
    "Logistic Regression": (LogisticRegression(max_iter=5000), {"m__C": [0.01, 0.1, 1, 10, 100]}),
}
tuned = {}
for n, (m, g) in grids.items():
    gs = GridSearchCV(Pipeline([("sc", StandardScaler()), ("m", m)]), g, cv=cv, scoring="recall_weighted" if False else "roc_auc", n_jobs=-1)
    gs.fit(Xtr, ytr); tuned[n] = gs
    R[f"tuned_{n}"] = {"params": gs.best_params_, "cv_auc": round(gs.best_score_, 4)}
best_name = max(tuned, key=lambda k: tuned[k].best_score_)
best = tuned[best_name].best_estimator_
pred = best.predict(Xte); proba = best.predict_proba(Xte)[:, 1]
R["best_model"] = best_name
R["final"] = dict(accuracy=accuracy_score(yte, pred), precision=precision_score(yte, pred),
                  recall=recall_score(yte, pred), f1=f1_score(yte, pred), roc_auc=roc_auc_score(yte, proba))
R["confusion"] = confusion_matrix(yte, pred).tolist()
R["report"] = classification_report(yte, pred, target_names=["Benign", "Malignant"])
print(R["final"], R["confusion"])

# Reduced-feature comparison (after removing collinear features)
red = Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, random_state=SEED, **{k[3:]: v for k, v in tuned[best_name].best_params_.items()}) if best_name.startswith("SVM")
                else LogisticRegression(max_iter=5000, C=tuned[best_name].best_params_["m__C"]))])
red.fit(Xtr[sel], ytr); pr2 = red.predict(Xte[sel])
R["reduced"] = dict(n=len(sel), accuracy=accuracy_score(yte, pr2), recall=recall_score(yte, pr2), f1=f1_score(yte, pr2),
                    roc_auc=roc_auc_score(yte, red.predict_proba(Xte[sel])[:, 1]))

# Plots
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
cmx = confusion_matrix(yte, pred)
sns.heatmap(cmx, annot=True, fmt="d", cmap="Blues", ax=ax[0], xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
ax[0].set_title(f"Confusion matrix - {best_name}"); ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("Actual")
for n, m in models.items():
    p = Pipeline([("sc", StandardScaler()), ("m", m)]).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    f, t, _ = roc_curve(yte, p); ax[1].plot(f, t, label=f"{n} ({roc_auc_score(yte, p):.3f})")
ax[1].plot([0, 1], [0, 1], "k--", lw=.8); ax[1].set_title("ROC curves (test set)"); ax[1].legend(fontsize=7)
ax[1].set_xlabel("False positive rate"); ax[1].set_ylabel("True positive rate")
plt.tight_layout(); plt.savefig("figures/06_confusion_roc.png", dpi=150); plt.close()

plt.figure(figsize=(8, 3.8)); r = res.set_index("Model")[["Test_Accuracy", "Recall", "F1", "ROC_AUC"]]
r.plot.bar(ax=plt.gca(), rot=15); plt.ylim(0.9, 1.01); plt.title("Model comparison on held-out test set"); plt.legend(fontsize=7, loc="lower right")
plt.tight_layout(); plt.savefig("figures/07_model_comparison.png", dpi=150); plt.close()

rf = RandomForestClassifier(n_estimators=300, random_state=SEED).fit(Xtr, ytr)
imp = pd.Series(rf.feature_importances_, index=feat).sort_values().tail(10)
R["top_importance"] = imp.sort_values(ascending=False).round(3).to_dict()
plt.figure(figsize=(7, 4)); imp.plot.barh(color="#2E6F95"); plt.title("Top 10 feature importances (Random Forest)")
plt.tight_layout(); plt.savefig("figures/08_feature_importance.png", dpi=150); plt.close()

# ---------- 5. Save ----------
joblib.dump({"model": best, "features": feat, "label_map": {0: "Benign", 1: "Malignant"},
             "medians": X.median().to_dict(), "ranges": {c: [float(X[c].min()), float(X[c].max())] for c in feat}}, "breast_cancer_model.joblib")
df.describe().T.round(3).to_csv("data_summary.csv")
json.dump(R, open("results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=1, default=str))
