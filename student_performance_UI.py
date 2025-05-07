import tkinter as tk
from tkinter import messagebox, ttk
import joblib
import numpy as np

# Load models
regression_model = joblib.load("student_score_predictor.pkl")
classification_model = joblib.load("student_pass_fail_predictor.pkl")

# Preprocess input
def preprocess_input(gender, study_hours, attendance, past_score,
                     internet, extra, parental_education):
    gender = 1 if gender == 'Male' else 0
    internet = 1 if internet == 'Yes' else 0
    extra = 1 if extra == 'Yes' else 0

    hs = 1 if parental_education == 'High School' else 0
  
    masters = 1 if parental_education == 'Masters' else 0
    phd = 1 if parental_education == 'PhD' else 0

    features = [gender, float(study_hours), float(attendance), float(past_score),
                internet, extra, hs,  masters, phd]
    return np.array(features).reshape(1, -1)

# Prediction
def predict_result():
    try:
        gender = gender_var.get()
        study_hours = study_hours_entry.get()
        attendance = attendance_entry.get()
        past_score = past_score_entry.get()
        internet = internet_var.get()
        extra = extra_var.get()
        parental_education = parental_edu_var.get()

        processed = preprocess_input(gender, study_hours, attendance, past_score,
                                     internet, extra, parental_education)

        final_score = regression_model.predict(processed)[0]
        pass_fail = classification_model.predict(processed)[0]
        status = "Pass" if pass_fail == 1 else "Fail"

        result_label.config(text=f"📊 Final Score: {final_score:.2f}\n📌 Status: {status}",
                            fg="green" if status == "Pass" else "red")

    except Exception as e:
        messagebox.showerror("Error", f"Something went wrong:\n{e}")

# --- UI Setup ---
root = tk.Tk()
root.title("🎓 Student Performance Predictor")
root.geometry("480x550")
root.config(bg="#f0f4f8")

title = tk.Label(root, text="📘 Predict Final Score & Result", font=("Helvetica", 18, "bold"), bg="#f0f4f8")
title.pack(pady=10)

frame = tk.Frame(root, bg="#f0f4f8")
frame.pack(pady=10)

# Helper for field creation
def add_field(label_text, widget, row):
    tk.Label(frame, text=label_text, font=("Helvetica", 12), bg="#f0f4f8").grid(row=row, column=0, sticky='e', padx=10, pady=8)
    widget.grid(row=row, column=1, padx=10, pady=8)

# Gender
gender_var = tk.StringVar(value="Male")
add_field("Gender:", ttk.Combobox(frame, textvariable=gender_var, values=["Male", "Female"], state="readonly"), 0)

# Study hours
study_hours_entry = tk.Entry(frame, font=("Helvetica", 12))
add_field("Study Hours/Week:", study_hours_entry, 1)

# Attendance
attendance_entry = tk.Entry(frame, font=("Helvetica", 12))
add_field("Attendance Rate (%):", attendance_entry, 2)

# Past scores
past_score_entry = tk.Entry(frame, font=("Helvetica", 12))
add_field("Past Exam Score:", past_score_entry, 3)

# Internet access
internet_var = tk.StringVar(value="Yes")
add_field("Internet at Home:", ttk.Combobox(frame, textvariable=internet_var, values=["Yes", "No"], state="readonly"), 4)

# Extracurriculars
extra_var = tk.StringVar(value="Yes")
add_field("Extracurricular Activities:", ttk.Combobox(frame, textvariable=extra_var, values=["Yes", "No"], state="readonly"), 5)

# Parental education
parental_edu_var = tk.StringVar(value="High School")
add_field("Parental Education Level:",
          ttk.Combobox(frame, textvariable=parental_edu_var,
                       values=["High School", "Bachelor's", "Masters", "PhD"],
                       state="readonly"), 6)

# Predict Button
tk.Button(root, text="🔮 Predict", command=predict_result,
          font=("Helvetica", 14, "bold"), bg="#4CAF50", fg="white",
          padx=10, pady=5).pack(pady=20)

# Output Label
result_label = tk.Label(root, text="", font=("Helvetica", 14), bg="#f0f4f8")
result_label.pack()

root.mainloop()
