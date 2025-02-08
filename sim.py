import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.linalg import inv
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize global variables
df = None
treated_df = None
control_df = None
inv_cov_matrix = None
features = ["Pain_Baseline", "Urgency_Baseline", "Frequency_Baseline"]

# Function to load CSV file
def load_data():
    global df, treated_df, control_df, inv_cov_matrix  # Make variables global
    filepath = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if not filepath:
        return
    
    try:
        df = pd.read_csv(filepath)

        # Convert Treatment_Status to categorical labels
        df["Treatment_Status"] = df["Treatment_Status"].astype(str)

        # Separate treated and control groups
        treated_df = df[df["Treatment_Status"] == "Treated"].reset_index(drop=True)
        control_df = df[df["Treatment_Status"] == "Not Yet Treated"].reset_index(drop=True)

        # Compute covariance matrix and its inverse
        cov_matrix = df[features].cov().values
        inv_cov_matrix = inv(cov_matrix)

        messagebox.showinfo("Success", "Data loaded successfully!")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load data: {e}")

# Function to find the best match based on Mahalanobis distance
def find_best_match(treated_row, control_group):
    best_match = None
    min_distance = float("inf")
    
    for _, control_row in control_group.iterrows():
        distance = mahalanobis(
            treated_row[features].values, control_row[features].values, inv_cov_matrix
        )
        if distance < min_distance:
            min_distance = distance
            best_match = control_row
    return best_match

# Function to perform BRSM Matching
def perform_matching():
    if df is None:
        messagebox.showerror("Error", "Please load data first.")
        return
    
    matches = []
    control_copy = control_df.copy()  # Avoid modifying original control data

    for _, treated_row in treated_df.iterrows():
        if control_copy.empty:
            messagebox.showinfo("Info", "No more control patients available for matching.")
            break

        best_match = find_best_match(treated_row, control_copy)
        if best_match is not None:
            matches.append((treated_row["Patient_ID"], best_match["Patient_ID"]))
            control_copy = control_copy[control_copy["Patient_ID"] != best_match["Patient_ID"]]  # Remove matched control

    matched_df = pd.DataFrame(matches, columns=["Treated_Patient_ID", "Matched_Control_Patient_ID"])
    display_results(matched_df)
    generate_boxplots(matched_df)

# Function to display matched pairs in GUI
def display_results(matched_df):
    for row in tree.get_children():
        tree.delete(row)
    for _, row in matched_df.iterrows():
        tree.insert("", "end", values=(row["Treated_Patient_ID"], row["Matched_Control_Patient_ID"]))

# Function to generate boxplots for balance checking
def generate_boxplots(matched_df):
    merged_df = treated_df.merge(matched_df, left_on="Patient_ID", right_on="Treated_Patient_ID")
    merged_df = merged_df.merge(control_df, left_on="Matched_Control_Patient_ID", right_on="Patient_ID",
                                suffixes=("_treated", "_control"))

    plt.figure(figsize=(12, 6))
    for i, covariate in enumerate(features):
        plt.subplot(1, 3, i + 1)
        sns.boxplot(data=merged_df[[f"{covariate}_treated", f"{covariate}_control"]])
        plt.title(f"Balance Check: {covariate}")
        plt.xticks([0, 1], ["Treated", "Control"])

    plt.tight_layout()
    plt.show()

# GUI Setup
root = tk.Tk()
root.title("Patient Data Viewer & BRSM Matching")

frame = tk.Frame(root)
frame.pack(pady=20)

tk.Button(frame, text="Load Data", command=load_data).grid(row=0, column=0, padx=10)
tk.Button(frame, text="Perform Matching", command=perform_matching).grid(row=0, column=1, padx=10)

# Treeview Table
tree = ttk.Treeview(root, columns=("Treated_Patient_ID", "Matched_Control_Patient_ID"), show="headings")
tree.heading("Treated_Patient_ID", text="Treated Patient ID")
tree.heading("Matched_Control_Patient_ID", text="Matched Control ID")
tree.pack(pady=20)

root.mainloop()
