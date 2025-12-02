from flask import Flask, render_template, request, jsonify, session
import os
import sys
import pandas as pd
from datetime import datetime

# Add Backend to sys.path absolutely
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "Backend")))

try:
    from Backend.Complaint_input_file import prepare_complaint
    from Backend.getpnr import extract_coach_train_seat_from_railrestro
    from Backend.Urgency_finder import check_urgency
    from Backend.Management_technical_split import classify_issue
    from Backend.police_complaint_details import police_complaint
    from Backend.sms_sender import send_sms, send_sms_to_multiple
    from Backend.voice_to_text import convert_voice_to_text  # <-- ONLY NEW MODULE
    from Backend.medical_finder import is_medical_issue
except ImportError as e:
    print(f"Import error: {e}")

app = Flask(__name__)
app.secret_key = "secret_key_for_session"

CSV_FILE = os.path.join("Storage Files", "Save_to_CSV.csv")
POLICE_CSV = os.path.join("Storage Files", "Police_id.csv")

os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=[
        "Timestamp", "PNR", "Train", "Coach", "Seat",
        "Complaint", "Urgency", "Issue_Type", "Police_Info", "Medical_Info"
    ]).to_csv(CSV_FILE, index=False)

def format_indian_number(num):
    num = str(num).strip()
    if not num.startswith("+91"):
        num = "+91" + num[-10:]
    return num


# ---------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat_page")
def chat_page():
    session["state"] = "ask_pnr"
    session["data"] = {}
    session["last_reply_sent"] = False
    return render_template("chat.html")


@app.route("/chat_default", methods=["GET"])
def chat_default():
    session["state"] = "ask_pnr"
    session["data"] = {}
    session["last_reply_sent"] = False
    return jsonify({
        "reply": "Welcome to AIRA!<br><b>🚆 Your Smart Railway Assistant</b><br>Do you have a PNR number?",
        "buttons": ["Yes", "No"]
    })


# ---------------------------------------------------------------
# VOICE TO TEXT ROUTE (ONLY NEW PART)
# ---------------------------------------------------------------
@app.route("/voice_to_text", methods=["POST"])
def voice_to_text():
    try:
        audio_base64 = request.json.get("audio", "")

        if not audio_base64:
            return jsonify({
                "success": False,
                "error": "No audio received."
            })

        result = convert_voice_to_text(audio_base64)
        return jsonify(result)

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})



# ---------------------------------------------------------------
# MAIN CHAT ROUTE (UNCHANGED)
# ---------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    try:
        user_msg = request.json.get("message", "").strip()
        state = session.get("state", "ask_pnr")
        data = session.get("data", {})
        history = session.get("history", [])
        bot_msg = ""
        buttons = ["Back", "Restart"]

        # Restart
        if user_msg.lower() in ["hi", "hello", "hey", "restart"]:
            session["state"] = "ask_pnr"
            session["data"] = {}
            session["history"] = []
            return jsonify({
                "reply": "Welcome to AIRA!<br><b>🚆 Your Smart Railway Assistant</b><br>Do you have a PNR number?",
                "buttons": ["Yes", "No", "Back", "Restart"]
            })

        # Back
        if user_msg.lower() == "back" and history:
            prev = history.pop()
            session["state"] = prev.get("state", "ask_pnr")
            session["data"] = prev.get("data", {})
            session["history"] = history
            return jsonify({
                "reply": prev.get("bot_msg", "Returning to previous step..."),
                "buttons": prev.get("buttons", ["Back", "Restart"])
            })

        # Save state history
        if state not in ["ask_pnr", "ask_another_complaint"]:
            history.append({
                "state": state,
                "data": data.copy(),
                "bot_msg": bot_msg,
                "buttons": buttons.copy()
            })
            session["history"] = history

        lower_msg = user_msg.lower()

        # ---------------- ASK PNR ----------------
        if state == "ask_pnr":
            if lower_msg == "yes":
                bot_msg = "Please enter your 10-digit PNR number:"
                session["state"] = "get_pnr"
            elif lower_msg == "no":
                bot_msg = "Please enter Train Number, Coach, and Seat separated by commas (e.g., 12345,A,12):"
                session["state"] = "get_train_coach_seat"
            else:
                bot_msg = "Please reply with Yes or No."
            buttons = ["Yes", "No", "Back", "Restart"]

        # ---------------- GET PNR ----------------
        elif state == "get_pnr":
            if len(user_msg) == 10 and user_msg.isdigit():
                train, coach, seat = extract_coach_train_seat_from_railrestro(user_msg) or ("", "", "")
                data.update({"pnr": user_msg, "train": train, "coach": coach, "seat": seat})
                session["data"] = data

                if train and coach and seat:
                    bot_msg = (
                        f"Fetched details:<br>Train: {train}<br>"
                        f"Coach: {coach}<br>Seat: {seat}<br>Is this correct?"
                    )
                    session["state"] = "confirm_details"
                else:
                    bot_msg = "Could not fetch details from PNR.<br>Please enter Train, Coach, Seat manually."
                    session["state"] = "get_train_coach_seat"
            else:
                bot_msg = "Invalid PNR. Please enter a valid 10-digit PNR:"
            buttons = ["Back", "Restart"]

        # ---------------- CONFIRM DETAILS ----------------
        elif state == "confirm_details":
            if lower_msg == "yes":
                bot_msg = "Please describe your complaint:"
                session["state"] = "get_complaint"
            elif lower_msg == "no":
                bot_msg = "Please enter Train, Coach, Seat manually (e.g., 12345,A,12):"
                session["state"] = "get_train_coach_seat"
            else:
                bot_msg = "Please confirm: Yes or No"
            buttons = ["Yes", "No", "Back", "Restart"]

        # ---------------- GET TRAIN COACH SEAT ----------------
        elif state == "get_train_coach_seat":
            try:
                train, coach, seat = [x.strip() for x in user_msg.split(",")]
                data.update({"train": train, "coach": coach, "seat": seat})
                session["data"] = data
                bot_msg = "Details saved ✅<br>Now, please describe your complaint:"
                session["state"] = "get_complaint"
            except Exception:
                bot_msg = "Invalid format. Please enter as Train,Coach,Seat (e.g., 12345,A,12):"
            buttons = ["Back", "Restart"]

        # ---------------- GET COMPLAINT ----------------
        elif state == "get_complaint":
            user_msg = user_msg.strip()
            important_keywords = {"help", "emergency", "urgent", "police", "fire", "ambulance"}
            word_count = len(user_msg.split())

            if len(user_msg) < 4:
                return jsonify({
                    "reply": "❌ Invalid complaint. Your message is too short.",
                    "buttons": ["Submit another complaint", "Back", "Restart"]
                })

            elif word_count == 1 and user_msg.lower() not in important_keywords:
                return jsonify({
                    "reply": "❌ Invalid complaint. Please provide more details about your issue.",
                    "buttons": ["Submit another complaint", "Back", "Restart"]
                })

            data["complaint"] = user_msg
            session["data"] = data

            urgency = check_urgency(user_msg)
            issue_type = classify_issue(user_msg)
            train, coach, seat = data.get("train"), data.get("coach"), data.get("seat")

            police_msg, medical_msg = "", ""

            sms_text = (
                f"🚨 URGENT ALERT 🚨\n"
                f"Train: {train}\nCoach: {coach}\nSeat: {seat}\n"
                f"Complaint: {user_msg}\nCategory: {issue_type}\nUrgency: {urgency}"
            )

            if urgency.lower() == "high":
                try:
                    df_police = pd.read_csv(POLICE_CSV)
                    police_row = df_police[
                        (df_police["Train Number"] == int(train)) &
                        (df_police["Compartment No"].astype(str).str.upper() == str(coach).upper())
                    ].iloc[0]

                    police_name = police_row["Officer Name"]
                    police_phone = format_indian_number(police_row["Officer Mobile No."])
                    send_sms(police_phone, sms_text)
                    police_msg = f"Message sent to Police Officer {police_name} "

                    if is_medical_issue(user_msg):
                        medical_numbers = [format_indian_number(police_row["Medical Team"])]
                        send_sms_to_multiple(medical_numbers, sms_text)
                        medical_msg = f"Message sent to Medical Team"

                except Exception as e:
                    print(f"Error fetching police/medical info: {e}")

            bot_msg = (
                f"Train: {train}<br>"
                f"Coach: {coach}<br>"
                f"Seat: {seat}<br>"
                f"Complaint: {user_msg}<br>"
                f"Category: {issue_type}<br>"
                f"Urgency: {urgency}<br>"
            )

            if police_msg:
                bot_msg += f"{police_msg}<br>"

            if medical_msg:
                bot_msg += f"{medical_msg}<br>"

            bot_msg += "✅ Complaint registered to Railways<br><br>Do you want to submit another complaint?"
            buttons = ["Yes", "No", "Back", "Restart", "Submit another complaint"]
            session["state"] = "ask_another_complaint"

            # Save complaint
            try:
                df_save = pd.read_csv(CSV_FILE) if os.stat(CSV_FILE).st_size > 0 else pd.DataFrame()
            except Exception:
                df_save = pd.DataFrame()

            new_row = pd.DataFrame([{
                "Timestamp": datetime.now(),
                "PNR": data.get("pnr", ""),
                "Train": train,
                "Coach": coach,
                "Seat": seat,
                "Complaint": user_msg,
                "Urgency": urgency,
                "Issue_Type": issue_type,
                "Police_Info": police_msg,
                "Medical_Info": medical_msg
            }])

            df_save = pd.concat([df_save, new_row], ignore_index=True)
            df_save.to_csv(CSV_FILE, index=False)

        # ---------------- ASK ANOTHER COMPLAINT ----------------
        elif state == "ask_another_complaint":
            if lower_msg in ["yes", "submit another complaint"]:
                bot_msg = "Please describe your complaint:"
                session["state"] = "get_complaint"
            elif lower_msg == "no":
                bot_msg = "Thank you for using AIRA! Have a safe journey 🚆"
                session["state"] = "ask_pnr"
                session["data"] = {}
                session["history"] = []
            else:
                bot_msg = "Please reply with Yes or No."

            buttons = ["Yes", "No", "Back", "Restart", "Submit another complaint"]

        session.modified = True
        return jsonify({"reply": bot_msg, "buttons": buttons})

    except Exception as e:
        print(f"Error in /chat: {e}")
        return jsonify({
            "reply": "An internal error occurred. Please try again.",
            "buttons": ["Back", "Restart"]
        }), 500


# ---------------------------------------------------------------
# RUN APP
# ---------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
