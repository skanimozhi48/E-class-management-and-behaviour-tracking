from flask import Flask, render_template, Response, jsonify, send_file
import cv2
import face_recognition
import numpy as np
import pandas as pd
import os
import time
# Import this at the top with the other imports
from datetime import timedelta

import threading
import winsound
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
import pytz


app = Flask(__name__)

KNOWN_FACES_DIR = "known_faces"
EXCEL_FILE = "attendance.xlsx"
EMAIL_SENDER = "kanimozhisit25@gmail.com"
EMAIL_PASSWORD = "wvrx rdzl bnxm gcpp"
EMAIL_RECEIVER = "kanisabari555@gmail.com"

IST = pytz.timezone("Asia/Kolkata")
PERIOD_TIMES = {
    "Period 1": "10:00:00",
    "Period 2": "11:00:00",
    "Period 3": "12:00:00",
    "Period 4": "14:00:00",
    "Period 5": "15:00:00",
    "Period 6": "16:00:00"
}

present_students = set()
last_unrecognized_time = None
beep_active = False
popup_messages = []
cap = None

def load_known_faces():
    known_encodings, known_names = [], []
    for person in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, person)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                image_path = os.path.join(person_dir, filename)
                try:
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    return np.array(known_encodings), known_names

known_encodings, known_names = load_known_faces()

def play_alert():
    global beep_active
    beep_active = True
    while beep_active:
        winsound.Beep(3500, 500)
        time.sleep(1)

def stop_alert():
    global beep_active
    beep_active = False

def initialize_attendance():
    columns = ["Name", "Date"] + [f"P{i}" for i in range(1, 7)] + [f"T{i}" for i in range(1, 7)]
    if os.path.exists(EXCEL_FILE):
        df = pd.read_excel(EXCEL_FILE, engine='openpyxl')
    else:
        df = pd.DataFrame(columns=columns)
    return df
def mark_attendance(name, period):
    now = datetime.now(IST)
    date = now.strftime("%Y-%m-%d")
    time_now = now.time()

    df = initialize_attendance()
    if not ((df["Name"] == name) & (df["Date"] == date)).any():
        new_row = {"Name": name, "Date": date}
        for i in range(1, 7):
            new_row[f"P{i}"] = "Absent"
            new_row[f"T{i}"] = "00:00:00"
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    period_index = int(period.split()[-1])
    period_start = datetime.strptime(PERIOD_TIMES[period], "%H:%M:%S").time()
    period_end = (datetime.strptime(PERIOD_TIMES[period], "%H:%M:%S") + pd.Timedelta(hours=1)).time()

    if period_start <= time_now <= period_end:
        already_present = df[
            (df["Name"] == name) & 
            (df["Date"] == date) & 
            (df[f"P{period_index}"] == "Present")
        ]
        if already_present.empty:
            current_time = now.strftime("%H:%M:%S")
            df.loc[(df["Name"] == name) & (df["Date"] == date), [f"P{period_index}", f"T{period_index}"]] = ["Present", current_time]
            df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')
            popup_messages.append(f"{name} marked present for {period} at {current_time}")
            present_students.add(name)
            return f"{name} marked present for {period}"

    return f"{name} not within {period} window, attendance not marked."


def update_absent_students():
    df = initialize_attendance()
    now = datetime.now(IST)
    date = now.strftime("%Y-%m-%d")
    time_now = now.time()

    # Add all known students for today if not already added
    for name in known_names:
        if not ((df["Name"] == name) & (df["Date"] == date)).any():
            new_row = {"Name": name, "Date": date}
            for i in range(1, 7):
                new_row[f"P{i}"] = "Absent"
                new_row[f"T{i}"] = "00:00:00"
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

    # Now update periods whose 1-hour window has passed but are still unmarked
    for index, row in df.iterrows():
        for period, period_start_str in PERIOD_TIMES.items():
            period_index = int(period.split()[-1])
            period_start = datetime.strptime(period_start_str, "%H:%M:%S").time()
            period_end = (datetime.strptime(period_start_str, "%H:%M:%S") + pd.Timedelta(hours=1)).time()

            if time_now > period_end and row[f"P{period_index}"] not in ["Present", "Absent"]:
                df.at[index, f"P{period_index}"] = "Absent"
                df.at[index, f"T{period_index}"] = "00:00:00"

    df.to_excel(EXCEL_FILE, index=False, engine='openpyxl')

def send_email_alert():
    df = initialize_attendance()
    now = datetime.now(IST)
    date = now.strftime("%Y-%m-%d")

    df_today = df[df["Date"] == date]
    present_students = df_today[df_today.iloc[:, 2:].eq("Present").any(axis=1)]["Name"].tolist()
    absent_students = list(set(known_names) - set(present_students))

    subject = "Attendance Report"
    message = f"Total Present: {len(present_students)}\nPresent Students: {', '.join(present_students)}\nAbsent Students: {', '.join(absent_students)}"

    try:
        msg = MIMEText(message)
        msg["Subject"] = subject
        msg["From"] = EMAIL_SENDER
        msg["To"] = EMAIL_RECEIVER
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_SENDER, EMAIL_RECEIVER, msg.as_string())
        server.quit()
        popup_messages.append("Email Sent Successfully")
        return "Email Sent Successfully"
    except Exception as e:
        return f"Error Sending Email: {e}"

def generate_frames(): 
    global cap, beep_active, last_unrecognized_time
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if not face_locations:
            if last_unrecognized_time is None:
                last_unrecognized_time = time.time()
            elif time.time() - last_unrecognized_time >= 10 and not beep_active:
                threading.Thread(target=play_alert, daemon=True).start()
                popup_messages.append("No face detected for 10s. Beep ON")
        else:
            last_unrecognized_time = None
            stop_alert()

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            name, color = "Unknown", (0, 0, 255)

            if True in matches:
                best_match_index = np.argmin(face_recognition.face_distance(known_encodings, face_encoding))
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    color = (0, 255, 0)
                    now = datetime.now(IST)
                    current_time = now.strftime("%H:%M:%S")

                    for period, period_time in PERIOD_TIMES.items():
                        period_index = int(period.split()[-1])
                        period_start = datetime.strptime(period_time, "%H:%M:%S").time()
                        period_end = (datetime.strptime(period_time, "%H:%M:%S") + pd.Timedelta(minutes=59, seconds=59)).time()

                        if period_start <= now.time() <= period_end:
                            df = initialize_attendance()
                            date = now.strftime("%Y-%m-%d")
                            already_marked = not df[
                                (df["Name"] == name) &
                                (df["Date"] == date) &
                                (df[f"P{period_index}"] == "Present")
                            ].empty
                            if not already_marked:
                                mark_attendance(name, period)
                            break  # Only check the current valid period

            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/download')
def download_file():
    return send_file(EXCEL_FILE, as_attachment=True)

@app.route('/stop')
def stop():
    global cap
    update_absent_students()
    send_email_alert()
    if cap:
        cap.release()
    cv2.destroyAllWindows()
    return jsonify({"message": "Webcam stopped and Email sent!"})

@app.route('/popup_messages')
def get_popup_messages():
    global popup_messages
    msgs = popup_messages.copy()
    popup_messages.clear()
    return jsonify({"messages": msgs})

if __name__ == '__main__':
    app.run(debug=True)
