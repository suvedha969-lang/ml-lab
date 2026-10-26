import pandas as pd
from sklearn.ensemble import IsolationForest

df = pd.read_csv("mplad_projects.csv")

print(df.shape)
print(df.columns)
print(df.head())

df["spent_percent"] = (df["spent_amount"] / df["approved_amount"]) * 100

print(df[["project_id", "approved_amount", "spent_amount", "spent_percent"]])
df["spending_vs_progress"] = df["spent_percent"] - df["progress_percent"]
print(df[["project_id", "spent_percent", "progress_percent", "spending_vs_progress"]])
df["risk_score"] = 0

df.loc[df["spending_vs_progress"] > 40, "risk_score"] += 50
df.loc[df["delay_days"] > 60, "risk_score"] += 30
df.loc[df["spent_percent"] > 90, "risk_score"] += 20
def get_risk_level(score):
    if score >= 61:
        return "High"
    elif score >= 31:
        return "Medium"
    else:
        return "Low"

df["risk_level"] = df["risk_score"].apply(get_risk_level)

print(df[["project_id", "risk_score", "risk_level"]])
features = df[["spent_percent", "progress_percent", "delay_days"]]

model = IsolationForest(contamination=0.2, random_state=42)

df["ai_anomaly"] = model.fit_predict(features)

df["ai_risk"] = df["ai_anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

print(df[["project_id", "risk_score", "ai_risk"]])
def get_risk_level(row):
    if row["ai_risk"] == "Anomaly" and row["risk_score"] >= 70:
        return "Critical"
    elif row["risk_score"] >= 70:
        return "High"
    elif row["risk_score"] >= 40:
        return "Medium"
    else:
        return "Low"

df["risk_level"] = df.apply(get_risk_level, axis=1)

print(df[["project_id", "risk_score", "ai_risk", "risk_level"]])