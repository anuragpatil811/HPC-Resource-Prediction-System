import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# STEP 1: CREATE SYNTHETIC DATA
# -------------------------------
data = []

for i in range(1000):
    file_size = np.random.randint(10, 50000)        # KB
    file_type = np.random.randint(1, 7)             # encoded type
    num_rows = np.random.randint(10, 100000)
    num_cols = np.random.randint(1, 200)

    # Simulated logic (IMPORTANT for realism)
    cpu_time = (
        file_size * 0.002 +
        num_rows * 0.0001 +
        file_type * 1.5 +
        np.random.uniform(0, 5)
    )

    memory = (
        file_size * 0.01 +
        num_cols * 3 +
        np.random.uniform(0, 20)
    )

    data.append([file_size, file_type, num_rows, num_cols, cpu_time, memory])

df = pd.DataFrame(data, columns=[
    "file_size", "file_type", "num_rows", "num_cols", "cpu_time", "memory"
])

# -------------------------------
# STEP 2: SPLIT DATA
# -------------------------------
X = df[["file_size", "file_type", "num_rows", "num_cols"]]
y_cpu = df["cpu_time"]
y_mem = df["memory"]

X_train, X_test, y_cpu_train, y_cpu_test = train_test_split(X, y_cpu, test_size=0.2)
_, _, y_mem_train, y_mem_test = train_test_split(X, y_mem, test_size=0.2)

# -------------------------------
# STEP 3: TRAIN MODELS
# -------------------------------
cpu_model = DecisionTreeRegressor(max_depth=5)
mem_model = DecisionTreeRegressor(max_depth=5)

cpu_model.fit(X_train, y_cpu_train)
mem_model.fit(X_train, y_mem_train)

# -------------------------------
# STEP 4: SAVE MODELS
# -------------------------------
joblib.dump(cpu_model, "cpu_model.pkl")
joblib.dump(mem_model, "mem_model.pkl")

print("Models created successfully!")