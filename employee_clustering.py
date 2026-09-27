import os
import heapq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

DATA_FILE = os.path.join(DATA_DIR, "employee_performance.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

keep_files = {
    "employee_grouped_results.csv",
    "cluster_summary.csv",
    "project_statistics.txt",
    "employee_dashboard.png",
    "employee_clusters_pca.png",
    "hierarchical_dendrogram.png"
}

for file_name in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, file_name)
    if os.path.isfile(file_path) and file_name not in keep_files:
        os.remove(file_path)

df = pd.read_csv(DATA_FILE)

required_columns = [
    "Employee_ID",
    "Department",
    "Education",
    "Job_Level",
    "City",
    "Age",
    "Experience",
    "Salary",
    "Performance_Score",
    "Working_Hours",
    "Training_Hours",
    "Satisfaction_Score"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    raise ValueError(
        "Missing columns: " + ", ".join(missing_columns)
    )

numeric_columns = [
    "Age",
    "Experience",
    "Salary",
    "Performance_Score",
    "Working_Hours",
    "Training_Hours",
    "Satisfaction_Score"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=numeric_columns).reset_index(drop=True)

print("\n" + "=" * 100)
print("EMPLOYEE PERFORMANCE DATA")
print("=" * 100)
print(df.to_string(index=False))

features = [
    "Experience",
    "Salary",
    "Performance_Score",
    "Working_Hours",
    "Training_Hours",
    "Satisfaction_Score"
]

X = df[features].values.astype(float)

mean_values = X.mean(axis=0)
std_values = X.std(axis=0)

std_values[std_values == 0] = 1

X_scaled = (X - mean_values) / std_values

n = len(X_scaled)

distance_matrix = np.zeros((n, n))

for i in range(n):
    differences = X_scaled[i + 1:] - X_scaled[i]
    distances = np.sqrt(np.sum(differences * differences, axis=1))
    distance_matrix[i, i + 1:] = distances
    distance_matrix[i + 1:, i] = distances

clusters = {i: [i] for i in range(n)}

active_clusters = set(range(n))

cluster_sizes = {i: 1 for i in range(n)}

heap = []

for i in range(n):
    for j in range(i + 1, n):
        heapq.heappush(heap, (distance_matrix[i, j], i, j))

merge_history = []

next_cluster_id = n

target_clusters = 3

while len(active_clusters) > target_clusters:
    distance, cluster_a, cluster_b = heapq.heappop(heap)

    if cluster_a not in active_clusters or cluster_b not in active_clusters:
        continue

    members_a = clusters[cluster_a]
    members_b = clusters[cluster_b]

    new_members = members_a + members_b

    new_cluster_id = next_cluster_id
    next_cluster_id += 1

    clusters[new_cluster_id] = new_members
    cluster_sizes[new_cluster_id] = len(new_members)

    active_clusters.remove(cluster_a)
    active_clusters.remove(cluster_b)
    active_clusters.add(new_cluster_id)

    merge_history.append(
        [
            cluster_a,
            cluster_b,
            distance,
            len(members_a),
            len(members_b)
        ]
    )

    for other_cluster in list(active_clusters):
        if other_cluster == new_cluster_id:
            continue

        members_other = clusters[other_cluster]

        total_distance = 0.0
        count = 0

        for a in new_members:
            for b in members_other:
                total_distance += distance_matrix[a, b]
                count += 1

        average_distance = total_distance / count

        heapq.heappush(
            heap,
            (
                average_distance,
                min(new_cluster_id, other_cluster),
                max(new_cluster_id, other_cluster)
            )
        )

cluster_ids = sorted(active_clusters)

cluster_labels = {}

for label, cluster_id in enumerate(cluster_ids, start=1):
    for employee_index in clusters[cluster_id]:
        cluster_labels[employee_index] = label

df["Cluster"] = [
    cluster_labels[i] for i in range(len(df))
]

output_file = os.path.join(
    OUTPUT_DIR,
    "employee_grouped_results.csv"
)

df.to_csv(output_file, index=False)

cluster_summary = df.groupby("Cluster").agg(
    Employees=("Employee_ID", "count"),
    Average_Age=("Age", "mean"),
    Average_Experience=("Experience", "mean"),
    Average_Salary=("Salary", "mean"),
    Average_Performance=("Performance_Score", "mean"),
    Average_Working_Hours=("Working_Hours", "mean"),
    Average_Training_Hours=("Training_Hours", "mean"),
    Average_Satisfaction=("Satisfaction_Score", "mean")
).reset_index()

cluster_summary_file = os.path.join(
    OUTPUT_DIR,
    "cluster_summary.csv"
)

cluster_summary.to_csv(
    cluster_summary_file,
    index=False
)

statistics_file = os.path.join(
    OUTPUT_DIR,
    "project_statistics.txt"
)

with open(statistics_file, "w", encoding="utf-8") as file:
    file.write("EMPLOYEE PERFORMANCE GROUPING\n")
    file.write("=" * 60 + "\n\n")
    file.write(f"Total Employees: {len(df)}\n")
    file.write(f"Number of Clusters: {target_clusters}\n")
    file.write("Algorithm: Agglomerative Hierarchical Clustering\n")
    file.write("Linkage: Average Linkage\n")
    file.write("Method: Unsupervised Machine Learning\n\n")

    file.write("Overall Statistics\n")
    file.write("-" * 60 + "\n")
    file.write(f"Average Age: {df['Age'].mean():.2f}\n")
    file.write(f"Average Experience: {df['Experience'].mean():.2f}\n")
    file.write(f"Average Salary: {df['Salary'].mean():.2f}\n")
    file.write(
        f"Average Performance Score: "
        f"{df['Performance_Score'].mean():.2f}\n"
    )
    file.write(
        f"Average Working Hours: "
        f"{df['Working_Hours'].mean():.2f}\n"
    )
    file.write(
        f"Average Training Hours: "
        f"{df['Training_Hours'].mean():.2f}\n"
    )
    file.write(
        f"Average Satisfaction Score: "
        f"{df['Satisfaction_Score'].mean():.2f}\n"
    )

    file.write("\nCluster Summary\n")
    file.write("-" * 60 + "\n")
    file.write(cluster_summary.to_string(index=False))

print("\n" + "=" * 100)
print("CLUSTER SUMMARY")
print("=" * 100)
print(cluster_summary.to_string(index=False))

pca_mean = X_scaled.mean(axis=0)
X_centered = X_scaled - pca_mean

u, s, vt = np.linalg.svd(
    X_centered,
    full_matrices=False
)

pca_data = X_centered @ vt[:2].T

plt.figure(figsize=(10, 7))

for cluster in sorted(df["Cluster"].unique()):
    mask = df["Cluster"] == cluster
    plt.scatter(
        pca_data[mask, 0],
        pca_data[mask, 1],
        label=f"Cluster {cluster}",
        alpha=0.7
    )

plt.title("Employee Clusters - PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

pca_file = os.path.join(
    OUTPUT_DIR,
    "employee_clusters_pca.png"
)

plt.savefig(pca_file, dpi=200)
plt.close()

merge_data = np.array(merge_history)

plt.figure(figsize=(12, 7))

if len(merge_data) > 0:
    merge_distances = merge_data[:, 2]
    merge_numbers = np.arange(1, len(merge_distances) + 1)

    plt.plot(
        merge_numbers,
        merge_distances,
        linewidth=2
    )

plt.title("Hierarchical Clustering Merge Distance")
plt.xlabel("Merge Step")
plt.ylabel("Average Linkage Distance")
plt.grid(alpha=0.3)
plt.tight_layout()

dendrogram_file = os.path.join(
    OUTPUT_DIR,
    "hierarchical_dendrogram.png"
)

plt.savefig(dendrogram_file, dpi=200)
plt.close()

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
axes[1, 0].set_title("Employees by Cluster")
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
axes[2, 2].tick_params(axis="x", rotation=45)

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

print("\nFiles created successfully:")
print(output_file)
print(cluster_summary_file)
print(statistics_file)
print(pca_file)
print(dendrogram_file)
print(dashboard_file)