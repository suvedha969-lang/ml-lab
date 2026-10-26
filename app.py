import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest
st.markdown("""
<style>

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
}

/* Dashboard title */
h1 {
    text-align: center;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-bottom: 35px;
}

/* Section titles */
h2, h3 {
    font-weight: 700 !important;
    margin-top: 30px !important;
}

/* Summary cards */
[data-testid="stMetric"] {
    background: white;
    border-radius: 15px;
    padding: 20px;
    border: 1px solid #e3e7ef;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    transition: 0.2s;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0px 8px 18px rgba(0,0,0,0.12);
}

/* Metric number */
[data-testid="stMetricValue"] {
    font-size: 32px !important;
    font-weight: 800 !important;
}

/* Upload box */
[data-testid="stFileUploader"] {
    border-radius: 12px;
}

/* Select boxes */
[data-baseweb="select"] {
    border-radius: 10px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 600;
    padding: 8px 20px;
}

</style>
""", unsafe_allow_html=True)
st.markdown("""
<div style="
    background:#172554;
    color:white;
    padding:25px;
    border-radius:12px;
    text-align:center;
    font-size:32px;
    font-weight:700;
    margin-bottom:25px;
">
    📊 MPLAD AI PROJECT MONITORING DASHBOARD
</div>
""", unsafe_allow_html=True)

st.write("APP.PY IS RUNNING")


uploaded_file = st.file_uploader(
    "Upload MPLAD Project CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_csv("mplad_projects.csv")

required_columns = [
    "project_id",
    "location",
    "project_type",
    "approved_amount",
    "spent_amount",
    "progress_percent",
    "delay_days"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Missing columns: " + ", ".join(missing_columns))
    st.stop()


# Calculate spending percentage
df["spent_percent"] = (df["spent_amount"] / df["approved_amount"]) * 100

# Compare spending with progress
df["spending_vs_progress"] = (
    df["spent_percent"] - df["progress_percent"]
)

# Rule-based risk score
df["risk_score"] = 0

df.loc[df["spending_vs_progress"] > 40, "risk_score"] += 50
df.loc[df["delay_days"] > 60, "risk_score"] += 30
df.loc[df["spent_percent"] > 90, "risk_score"] += 20

# AI anomaly detection
features = df[["spent_percent", "progress_percent", "delay_days"]]

model = IsolationForest(
    contamination=0.2,
    random_state=42
)

df["ai_anomaly"] = model.fit_predict(features)

df["ai_risk"] = df["ai_anomaly"].map({
    1: "Normal",
    -1: "Anomaly"
})

# Final risk level
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
def get_risk_reason(row):
    reasons = []

    if row["spending_vs_progress"] > 40:
        reasons.append("High spending vs progress")

    if row["delay_days"] > 60:
        reasons.append("Project delay")

    if row["spent_percent"] > 90:
        reasons.append("High expenditure")

    if row["ai_risk"] == "Anomaly":
        reasons.append("AI anomaly")

    if not reasons:
        return "No major risk"

    return ", ".join(reasons)


df["risk_reason"] = df.apply(get_risk_reason, axis=1)


# Dashboard numbers
st.subheader("Project Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Projects", len(df))
col2.metric("High Risk", len(df[df["risk_level"] == "High"]))
col3.metric("Critical", len(df[df["risk_level"] == "Critical"]))
col4.metric("Anomalies", len(df[df["ai_risk"] == "Anomaly"]))
critical_projects = df[df["risk_level"] == "Critical"]

if len(critical_projects) > 0:
    st.error(
        f"🚨 ALERT: {len(critical_projects)} critical-risk project(s) detected!"
    )
else:
    st.success("✅ No critical-risk projects detected.")

# Project table
st.subheader("Risk Level Distribution")
risk_counts=df["risk_level"].value_counts()
st.bar_chart(risk_counts)
st.subheader("Project Risk Score")

risk_chart = df.set_index("project_id")["risk_score"]

st.bar_chart(risk_chart)
st.subheader("Spending vs Work progress")
chart_data=df.set_index("project_id")[["spent_percent","progress_percent"]]
st.line_chart(chart_data)
st.subheader("Filter Projects")
locations=["All"]+sorted(df["location"].unique().tolist())
selected_location=st.selectbox("Select Location",locations)
if selected_location!="All":
    filtered_df=df[df["location"]==selected_location]
else:
    filtered_df=df
risk_levels=["All"]+sorted(df["risk_level"].unique().tolist())
selected_risk=st.selectbox("Select Risk Level",risk_levels)
if selected_risk!="All":
    filtered_df=filtered_df[filtered_df["risk_level"]==selected_risk] 
search_project=st.text_input("Search Project ID")
if search_project:
    filtered_df=filtered_df[filtered_df["project_id"].str.contains(search_project,case=False,na=False)]

st.subheader("Project Risk Analysis")



st.dataframe(
    filtered_df[
        [
            "project_id",
            "location",
            "project_type",
            "approved_amount",
            "spent_amount",
            "progress_percent",
            "delay_days",
            "risk_score",
            "ai_risk",
            "risk_level",
            "risk_reason"
        ]
    ].style.map(
        lambda value: (
            "background-color: #ffcccc"
            if value == "Critical"
            else "background-color: #ffe0b3"
            if value == "High"
            else "background-color: #fff2cc"
            if value == "Medium"
            else "background-color: #ccffcc"
            if value == "Low"
            else ""
        ),
        subset=["risk_level"]
    )
)
st.subheader("Project Details")

if len(filtered_df) > 0:

    selected_project = st.selectbox(
        "Select a Project",
        filtered_df["project_id"].tolist()
    )

    project = filtered_df[
        filtered_df["project_id"] == selected_project
    ].iloc[0]
    

    st.write("### Project:", project["project_id"])

    st.write("**Location:**", project["location"])
    st.write("**Project Type:**", project["project_type"])
    st.write("**Approved Amount:** ₹", project["approved_amount"])
    st.write("**Spent Amount:** ₹", project["spent_amount"])
    st.write("**Work Progress:**", project["progress_percent"], "%")
    st.write("**Delay:**", project["delay_days"], "days")
    st.write("**Risk Score:**", project["risk_score"])
    st.write("**AI Result:**", project["ai_risk"])
    st.write("**Risk Level:**", project["risk_level"])

    st.write("### Why was this project flagged?")

    if project["spending_vs_progress"] > 40:
        st.warning(
            "High spending compared with physical work progress."
        )

    if project["delay_days"] > 60:
        st.warning(
            "Project has a significant completion delay."
        )

    if project["spent_percent"] > 90:
        st.warning(
            "More than 90% of the approved amount has been spent."
        )

    if project["ai_risk"] == "Anomaly":
        st.warning(
            "AI detected this project as unusual compared with other projects."
        )

    if project["risk_score"] < 40 and project["ai_risk"] == "Normal":
        st.success("No major risk indicators detected.")
st.subheader("🚨 Priority Projects")

high_risk_df = df[
    df["risk_level"].isin(["High", "Critical"])
]

if len(high_risk_df) > 0:

    selected_priority = st.selectbox(
        "Select a Priority Project",
        high_risk_df["project_id"].tolist()
    )

    project = high_risk_df[
        high_risk_df["project_id"] == selected_priority
    ].iloc[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Risk Level", project["risk_level"])
    col2.metric("Risk Score", project["risk_score"])
    col3.metric("AI Status", project["ai_risk"])

    st.write("**Location:**", project["location"])
    st.write("**Project Type:**", project["project_type"])
    st.write("**Risk Reason:**", project["risk_reason"])

else:
    st.success("No priority projects detected.")