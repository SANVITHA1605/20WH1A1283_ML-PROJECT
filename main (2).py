import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_log_error
import tkinter as tk
from tkinter import ttk


# Load the data into a Pandas dataframe
train_df = pd.read_csv('C:\\Users\\sainath\\Desktop\\ml\\train.csv')

# Separate the features (inputs) and targets
features = ['hour', 'minute', 'x', 'y', 'dir_EB', 'dir_NB']
target = 'congestion'
# Split the data into training and testing sets
train_features, val_features, train_target, val_target = train_test_split(train_df[features], train_df[target], test_size=0.2, random_state=42)

# train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(train_features, train_target)
y_pred=model.predict(val_features)

# Calculate the mean column-wise root mean squared logarithmic error
#msle = np.sqrt(mean_squared_log_error(y_test, y_pred, multioutput='raw_values')).mean()

# Create a GUI using tkinter
root = tk.Tk()
root.title("Air Pollution Measurements Prediction")
root.geometry("1000x1000")


# Load the background image
# bg_image = Image.open(os.path.join(os.path.dirname(file), 'C:\\Users\\DELL\\OneDrive\\Desktop\\mlproject\\backg.png'))
#bg_image = Image.open(os.path.join(os.path.dirname(_file_), 'C:\\Users\\DELL\\OneDrive\\Desktop\\mlproject\\backg.png'))

#bg_image = bg_image.resize((1500, 1000), Image.LANCZOS)
#bg_photo = ImageTk.PhotoImage(bg_image)
#bg_label = tk.Label(root, image=bg_photo)
#bg_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a frame for the input widgets
input_frame = ttk.Frame(root, borderwidth=0, relief="groove")
input_frame.place(relx=0.0, rely=0.0, anchor="center", width=0, height=0)





hour_label = tk.Label(window, text="Hour:")
hour_label.pack()
hour_entry = tk.Entry(window)
hour_entry.pack()

minute_label = tk.Label(window, text="Minute:")
minute_label.pack()
minute_entry = tk.Entry(window)
minute_entry.pack()

x_label = tk.Label(window, text="X-coordinate:")
x_label.pack()
x_entry = tk.Entry(window)
x_entry.pack()

y_label = tk.Label(window, text="Y-coordinate:")
y_label.pack()
y_entry = tk.Entry(window)
y_entry.pack()

direction_label = tk.Label(window, text="Direction:")
direction_label.pack()
direction_entry = tk.Entry(window)
direction_entry.pack()

# define a function to get the input values and display the predicted congestion
def predict_congestion():
    # get the input values
    hour = int(hour_entry.get())
    minute = int(minute_entry.get())
    x = float(x_entry.get())
    y = float(y_entry.get())
    direction = direction_entry.get()

    # encode the direction value using one-hot encoding
    if direction == "EB":
        dir_EB = 1
        dir_NB = 0
        dir_SB = 0
        dir_WB = 0
    elif direction == "NB":
        dir_EB = 0
        dir_NB = 1
        dir_SB = 0
        dir_WB = 0
    elif direction == "SB":
        dir_EB = 0
        dir_NB = 0
        dir_SB = 1
        dir_WB = 0
    elif direction == "WB":
        dir_EB = 0
        dir_NB = 0
        dir_SB = 0
        dir_WB = 1
    else:
        messagebox.showerror("Error", "Invalid direction")

    # predict the congestion value
    prediction = model.predict([[hour, minute, x, y, dir_EB, dir_NB]])

    # display the predicted congestion value
    messagebox.showinfo("Prediction", f"Predicted congestion value: {prediction[0]}")

# add a button to make the prediction
predict_button = tk.Button(window, text="Predict congestion", command=predict_congestion)
predict_button.pack()

# start the GUI window
window.mainloop()