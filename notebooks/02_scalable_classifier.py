import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Configuration
BASELINE_DIMS = 100000
DIMENSION_RANGE = [100, 200, 500, 700, 1000]
CHOSEN_D_FOR_SPEED = 100
CV_FOLDS = 5
GRID_PARAM_COUNT = 3
TOTAL_RUNS = CV_FOLDS * GRID_PARAM_COUNT

print(f"Baseline Dims: {BASELINE_DIMS}")
print(f"Scaled Dims:   {CHOSEN_D_FOR_SPEED}")
print(f"Test Range:    {DIMENSION_RANGE}")

# 1. Loading Data
full_dataset = load_dataset("ai4privacy/pii-masking-300k")
train_df = full_dataset['train'].to_pandas()
val_df = full_dataset['validation'].to_pandas()

def create_label(row):
    col = 'privacy_mask' if 'privacy_mask' in row else 'span_labels'
    try: return 1 if len(row[col]) > 0 else 0
    except: return 0

train_df['label'] = train_df.apply(create_label, axis=1)
val_df['label'] = val_df.apply(create_label, axis=1)

def balance_split(df):
    if df[df['label']==0].shape[0] < 100:
        df_safe = df.sample(frac=1.0, random_state=42).copy()
        if 'target_text' in df_safe.columns: df_safe['source_text'] = df_safe['target_text']
        df_safe['label'] = 0
        return pd.concat([df, df_safe], ignore_index=True)
    return df

train_df = balance_split(train_df)
val_df = balance_split(val_df)
y_train = train_df['label']
y_test = val_df['label']

print(f"   Train Rows: {len(train_df)}")
print(f"   Test Rows:  {len(val_df)}")

# Class Balance Check
counts = train_df['label'].value_counts()
print(f"   Class Counts:\n{counts}")
plt.figure(figsize=(6, 4))
counts.plot(kind='bar', color=['skyblue', 'salmon'])
plt.title("Distribution of Safe (0) vs. Sensitive (1)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

# 2. Vectorization (High-Dim Baseline)
vectorizer = TfidfVectorizer(max_features=BASELINE_DIMS)
X_train_high = vectorizer.fit_transform(train_df['source_text'].astype(str))
X_test_high = vectorizer.transform(val_df['source_text'].astype(str))

# Baseline accuracy check
clf_base = LogisticRegression(solver='saga', max_iter=100, n_jobs=-1)
print(f"   Pre-calculating Baseline Accuracy (CV={CV_FOLDS})...")
base_acc = cross_val_score(clf_base, X_train_high, y_train, cv=CV_FOLDS).mean()
print(f"   Baseline CV Acc: {base_acc:.4f}")

# 3. Scaling Analysis (Sparse JL Transform)
n_samples = len(train_df)
eps = 0.5
min_dim_theoretical = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)
print(f"   Theoretical Min Dim (eps={eps}): {min_dim_theoretical}")
print(f"   Our Choice: {CHOSEN_D_FOR_SPEED}")

results_scaling = []

for d in DIMENSION_RANGE:
    print(f"   Testing d={d}...", end="")
    start_d = time.time()

    sparse = SparseRandomProjection(n_components=d, density='auto', random_state=42)
    X_sparse = sparse.fit_transform(X_train_high)

    s_acc = cross_val_score(LogisticRegression(solver='saga', max_iter=50, n_jobs=-1),
                            X_sparse, y_train, cv=CV_FOLDS).mean()

    time_taken = time.time() - start_d

    print(f" Done. (Sparse Acc: {s_acc:.3f} | Time: {time_taken:.2f}s)")
    results_scaling.append({'d': d, 'Sparse': s_acc, 'Time': time_taken})

# Plot Accuracy vs Dimension
res_df = pd.DataFrame(results_scaling)
plt.figure(figsize=(10, 6))
plt.plot(res_df['d'], res_df['Sparse'], 's-', color='orange', label='Sparse JL')
plt.axhline(y=base_acc, color='r', linestyle='--', label=f'Baseline (D={BASELINE_DIMS})')

plt.title(f"Randomized Scaling: Accuracy vs Dimension (Target d={CHOSEN_D_FOR_SPEED})")
plt.xlabel("Projection Dimension (d)")
plt.ylabel("CV Accuracy")
plt.legend()
plt.grid(True)
plt.show()

print("\n   [Time Analysis per Dimension]")
print(res_df[['d', 'Time', 'Sparse']])

# 4. Asymptotic Analysis
scaler_sparse = SparseRandomProjection(n_components=CHOSEN_D_FOR_SPEED, density='auto', random_state=42)
X_train_low = scaler_sparse.fit_transform(X_train_high)

def get_curve(model, X, y):
    sizes, _, _, fit_times, _ = learning_curve(
        model, X, y, cv=CV_FOLDS, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), return_times=True
    )
    return sizes, np.mean(fit_times, axis=1)

print("   Generating Learning Curves...")
sizes, t_base = get_curve(LogisticRegression(solver='saga', max_iter=50), X_train_high, y_train)
_, t_scaled = get_curve(LogisticRegression(solver='saga', max_iter=50), X_train_low, y_train)

t_base_total = t_base * TOTAL_RUNS
t_scaled_total = t_scaled * TOTAL_RUNS

# Plot Asymptotic Time Complexity
plt.figure(figsize=(10, 6))
plt.plot(sizes, t_base_total, 'o-', color='r', label=f'Baseline Pipeline (Grid Search)')
plt.plot(sizes, t_scaled_total, 'o-', color='g', label=f'Sparse Scaled Pipeline (Grid Search)')
plt.title(f"Asymptotic Analysis: Total Pipeline Time vs N (15 Fits)")
plt.xlabel("Training Samples (N)")
plt.ylabel("Total Pipeline Time (s)")
plt.legend()
plt.grid(True)
plt.show()

# 5. Final Comparison: Baseline vs Scaled
param_grid = {'C': [0.1, 1, 10]}

def run_full_pipeline(name, X_input, y_train, X_test_input, y_test, scaler=None):
    print(f"   Running Pipeline: {name}...")
    start_total = time.time()

    if scaler:
        X_tr = scaler.fit_transform(X_input)
        X_te = scaler.transform(X_test_input)
    else:
        X_tr = X_input
        X_te = X_test_input

    proj_time = time.time() - start_total

    print(f"     -> Tuning (Grid Search)...")
    clf = LogisticRegression(solver='saga', max_iter=100, n_jobs=-1)
    grid = GridSearchCV(clf, param_grid, cv=CV_FOLDS, scoring='accuracy', n_jobs=-1)
    grid.fit(X_tr, y_train)

    total_time = time.time() - start_total
    best = grid.best_estimator_
    tr_acc = accuracy_score(y_train, best.predict(X_tr))
    y_pred_test = best.predict(X_te)
    te_acc = accuracy_score(y_test, y_pred_test)

    print(f"     -> Total Time: {total_time:.2f}s (Proj: {proj_time:.2f}s)")
    return total_time, tr_acc, te_acc, y_test, y_pred_test

# Run Baseline
base_time, base_tr, base_te, _, _ = run_full_pipeline(
    "Baseline (100k)", X_train_high, y_train, X_test_high, y_test, scaler=None
)

# Run Scaled Pipeline
sparse_scaler = SparseRandomProjection(n_components=CHOSEN_D_FOR_SPEED, density='auto',
                                     dense_output=False, random_state=42)
sparse_time, sparse_tr, sparse_te, y_true_final, y_pred_final = run_full_pipeline(
    "Sparse JL", X_train_high, y_train, X_test_high, y_test, scaler=sparse_scaler
)

results = {
    'Model': ['Baseline (100k)', f'Sparse JL ({CHOSEN_D_FOR_SPEED})'],
    'Total Pipeline Time (s)': [base_time, sparse_time],
    'Train Accuracy': [base_tr, sparse_tr],
    'Test Accuracy': [base_te, sparse_te]
}
df_res = pd.DataFrame(results)
print(df_res)
print(f"\nSpeedup: {base_time/sparse_time:.2f}x")

# Failure Analysis (Confusion Matrix)
cm = confusion_matrix(y_true_final, y_pred_final)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Safe', 'Risky'])
fig, ax = plt.subplots(figsize=(5, 5))
disp.plot(cmap='Blues', ax=ax)
plt.title(f"Confusion Matrix (Sparse JL, d={CHOSEN_D_FOR_SPEED})")
plt.show()

# 6. 3D Visualization
viz_pca = PCA(n_components=3)
idx = np.random.choice(X_train_high.shape[0], 2000, replace=False)
X_sample_sparse = sparse_scaler.transform(X_train_high[idx])
X_viz = viz_pca.fit_transform(X_sample_sparse.toarray())
y_viz = y_train.iloc[idx].values

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
colors = ['blue' if lbl==0 else 'red' for lbl in y_viz]
ax.scatter(X_viz[:,0], X_viz[:,1], X_viz[:,2], c=colors, alpha=0.5, s=15)
ax.set_title(f"3D Separability (Sparse JL, d={CHOSEN_D_FOR_SPEED})")
ax.set_xlabel("PC 1")
ax.set_ylabel("PC 2")
ax.set_zlabel("PC 3")
plt.show()
