import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

RESULT_FILE = os.path.join(
    OUTPUT_DIR,
    "employee_grouped_results.csv"
)

df = pd.read_csv(RESULT_FILE)

print("\n" + "=" * 100)
print("EMPLOYEE GROUPED DATA")
print("=" * 100)
print(df.to_string(index=False))

print("\n" + "=" * 100)
print("PROJECT SUMMARY")
print("=" * 100)

print(f"Total Employees: {len(df)}")
print(f"Total Clusters: {df['Cluster'].nunique()}")
print(f"Average Salary: ₹{df['Salary'].mean():,.2f}")
print(
    f"Average Performance: "
    f"{df['Performance_Score'].mean():.2f}"
)
print(
    f"Average Experience: "
    f"{df['Experience'].mean():.2f}"
)
print(
    f"Average Working Hours: "
    f"{df['Working_Hours'].mean():.2f}"
)
print(
    f"Average Training Hours: "
    f"{df['Training_Hours'].mean():.2f}"
)
print(
    f"Average Satisfaction: "
    f"{df['Satisfaction_Score'].mean():.2f}"
)

cluster_summary = df.groupby("Cluster").agg(
    Employees=("Employee_ID", "count"),
    Average_Experience=("Experience", "mean"),
    Average_Salary=("Salary", "mean"),
    Average_Performance=("Performance_Score", "mean"),
    Average_Working_Hours=("Working_Hours", "mean"),
    Average_Training_Hours=("Training_Hours", "mean"),
    Average_Satisfaction=("Satisfaction_Score", "mean")
).reset_index()

print("\n" + "=" * 100)
print("CLUSTER SUMMARY")
print("=" * 100)
print(cluster_summary.to_string(index=False))

department_summary = df["Department"].value_counts().reset_index()

department_summary.columns = [
    "Department",
    "Employees"
]

print("\n" + "=" * 100)
print("DEPARTMENT SUMMARY")
print("=" * 100)
print(department_summary.to_string(index=False))

fig, axes = plt.subplots(
    3,
    3,
    figsize=(16, 12)
)

fig.suptitle(
    "Employee Performance Grouping Dashboard",
    fontsize=20,
    fontweight="bold"
)

axes[0, 0].text(
    0.5,
    0.5,
    str(len(df)),
    ha="center",
    va="center",
    fontsize=35,
    fontweight="bold"
)
axes[0, 0].set_title("Total Employees")
axes[0, 0].axis("off")

axes[0, 1].text(
    0.5,
    0.5,
    f"₹{df['Salary'].mean():,.0f}",
    ha="center",
    va="center",
    fontsize=28,
    fontweight="bold"
)
axes[0, 1].set_title("Average Salary")
axes[0, 1].axis("off")

axes[0, 2].text(
    0.5,
    0.5,
    f"{df['Performance_Score'].mean():.2f}",
    ha="center",
    va="center",
    fontsize=32,
    fontweight="bold"
)
axes[0, 2].set_title("Average Performance")
axes[0, 2].axis("off")

cluster_counts = df["Cluster"].value_counts().sort_index()

axes[1, 0].bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

axes[1, 0].set_title("Employee Distribution by Cluster")
axes[1, 0].set_xlabel("Cluster")
axes[1, 0].set_ylabel("Employees")

cluster_performance = df.groupby(
    "Cluster"
)["Performance_Score"].mean()

axes[1, 1].bar(
    cluster_performance.index.astype(str),
    cluster_performance.values
)

axes[1, 1].set_title("Performance by Cluster")
axes[1, 1].set_xlabel("Cluster")
axes[1, 1].set_ylabel("Performance Score")

cluster_salary = df.groupby(
    "Cluster"
)["Salary"].mean()

axes[1, 2].bar(
    cluster_salary.index.astype(str),
    cluster_salary.values
)

axes[1, 2].set_title("Average Salary by Cluster")
axes[1, 2].set_xlabel("Cluster")
axes[1, 2].set_ylabel("Salary")

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    axes[2, 0].scatter(
        cluster_data["Experience"],
        cluster_data["Performance_Score"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

axes[2, 0].set_title("Experience vs Performance")
axes[2, 0].set_xlabel("Experience")
axes[2, 0].set_ylabel("Performance")
axes[2, 0].legend()

for cluster in sorted(df["Cluster"].unique()):
    cluster_data = df[df["Cluster"] == cluster]

    axes[2, 1].scatter(
        cluster_data["Salary"],
        cluster_data["Performance_Score"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )

axes[2, 1].set_title("Salary vs Performance")
axes[2, 1].set_xlabel("Salary")
axes[2, 1].set_ylabel("Performance")
axes[2, 1].legend()

department_counts = df["Department"].value_counts()

axes[2, 2].bar(
    department_counts.index,
    department_counts.values
)

axes[2, 2].set_title("Employees by Department")
axes[2, 2].set_xlabel("Department")
axes[2, 2].set_ylabel("Employees")
axes[2, 2].tick_params(
    axis="x",
    rotation=45
)

plt.tight_layout()

dashboard_file = os.path.join(
    OUTPUT_DIR,
    "employee_dashboard.png"
)

plt.savefig(
    dashboard_file,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("\nDashboard created successfully:")
print(dashboard_file)