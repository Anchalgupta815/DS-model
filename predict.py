"""Use the trained model to predict Benign / Malignant.
Usage:
  python predict.py --demo            # predicts 4 sample patients from the dataset
  python predict.py --csv input.csv   # CSV with the 30 feature columns -> adds prediction columns
  python predict.py                   # interactive: type values (press Enter to use the dataset median)
"""
import sys, joblib, pandas as pd

art = joblib.load("breast_cancer_model.joblib")
model, feats, labels = art["model"], art["features"], art["label_map"]

def predict_df(df):
    X = df[feats]
    p = model.predict_proba(X)[:, 1]
    out = df.copy()
    out["prediction"] = [labels[int(v >= 0.5)] for v in p]
    out["malignant_probability"] = p.round(4)
    return out

if "--demo" in sys.argv:
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True).frame
    sample = d.iloc[[0, 19, 100, 500]]
    res = predict_df(sample.drop(columns="target"))
    res["actual"] = sample["target"].map({0: "Malignant", 1: "Benign"}).values
    print(res[["prediction", "malignant_probability", "actual"]])
elif "--csv" in sys.argv:
    path = sys.argv[sys.argv.index("--csv") + 1]
    res = predict_df(pd.read_csv(path)); res.to_csv("predictions.csv", index=False)
    print(res[["prediction", "malignant_probability"]]); print("Saved to predictions.csv")
else:
    row = {}
    print("Enter tumour measurements (Enter = dataset median):")
    for f in feats:
        lo, hi = art["ranges"][f]
        v = input(f"  {f} [{lo:.4g} - {hi:.4g}, median {art['medians'][f]:.4g}]: ").strip()
        row[f] = float(v) if v else art["medians"][f]
    r = predict_df(pd.DataFrame([row])).iloc[0]
    print(f"\nPrediction: {r['prediction']}  (probability malignant = {r['malignant_probability']:.2%})")
