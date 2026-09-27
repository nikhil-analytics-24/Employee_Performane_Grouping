# Employee Performance Grouping Using Hierarchical Clustering

## Project Overview

Employee Performance Grouping Using Hierarchical Clustering is an Unsupervised Machine Learning project designed to analyze and group employees according to similarities in their professional and performance-related characteristics.

The project uses Agglomerative Hierarchical Clustering to divide employees into meaningful groups without requiring predefined labels.

The system generates employee data, validates and preprocesses the dataset, standardizes numerical features, performs hierarchical clustering, analyzes the resulting groups and presents the results through multiple visualizations and an advanced analytics dashboard.

---

## Project Title

**Employee Performance Grouping Using Hierarchical Clustering**

---

## Machine Learning Type

**Unsupervised Machine Learning**

---

## Algorithm Used

**Agglomerative Hierarchical Clustering**

Linkage Method:

**Average Linkage**

Number of Clusters:

**3**

---

## Project Objectives

The main objectives of this project are:

1. Generate an employee performance dataset.
2. Analyze employee-related characteristics.
3. Validate the dataset.
4. Select relevant numerical features.
5. Standardize the machine learning features.
6. Calculate similarities between employees.
7. Apply Agglomerative Hierarchical Clustering.
8. Divide employees into meaningful groups.
9. Analyze cluster characteristics.
10. Generate statistical reports.
11. Create multiple data visualizations.
12. Develop an advanced employee analytics dashboard.

---

## Dataset

The project currently uses a generated dataset containing:

**500 Employee Records**

Each employee record contains the following attributes:

| Feature | Description |
|---|---|
| Employee_ID | Unique employee identification number |
| Department | Employee department |
| Education | Educational qualification |
| Job_Level | Employee job level |
| City | Employee city |
| Age | Employee age |
| Experience | Professional experience in years |
| Salary | Employee annual salary |
| Performance_Score | Employee performance score |
| Working_Hours | Average working hours |
| Training_Hours | Training hours |
| Satisfaction_Score | Employee satisfaction score |
| Cluster | Machine learning cluster |

---

## Machine Learning Features

The clustering algorithm uses the following numerical features:

- Experience
- Salary
- Performance Score
- Working Hours
- Training Hours
- Satisfaction Score

Categorical fields such as Department, Education, Job Level and City are used for additional analysis and dashboard visualization.

---

## Data Preprocessing

Before applying clustering, the following preprocessing steps are performed:

1. Dataset generation
2. Missing value checking
3. Duplicate record checking
4. Numerical feature selection
5. Feature standardization
6. Euclidean distance calculation

Standardization is performed so that features with different numerical scales can be compared fairly.

---

## Hierarchical Clustering

The project implements Agglomerative Hierarchical Clustering.

The algorithm initially treats every employee as an individual cluster.

Similar clusters are then progressively merged according to their average distance.

The merging process continues until the required number of clusters is reached.

The project uses:

**Number of Clusters = 3**

---

## Cluster Analysis

The generated clusters are analyzed using:

- Average Experience
- Average Salary
- Average Performance Score
- Average Working Hours
- Training Hours
- Satisfaction Score
- Number of Employees

The cluster summary is automatically generated in:

`output/cluster_summary.csv`

---

## Project Workflow

```text
Employee Dataset Generation
            ↓
Data Validation
            ↓
Feature Selection
            ↓
Feature Standardization
            ↓
Distance Calculation
            ↓
Agglomerative Hierarchical Clustering
            ↓
Cluster Assignment
            ↓
Cluster Analysis
            ↓
Visualization
            ↓
Dashboard Generation
            ↓
Project Report