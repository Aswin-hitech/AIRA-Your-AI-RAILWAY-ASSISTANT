import os
import csv
import pandas as pd

from Complaint_input_file import prepare_complaint
from Management_technical_split import classify_issue
from Urgency_finder import check_urgency
from police_complaint_details import police_complaint
from sms_sender import send_sms
from medical_finder import medical_complaint, is_medical_issue

# Load police + medical data globally
police_df = pd.read_csv("Police_id.csv")


def save_complaint(data, category, urgency, filename="Save_to_CSV.csv"):
    """Save complaint details into CSV."""
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="", encoding="utf-8") as file:
        fieldnames = ["PNR", "Train", "Coach", "Seat", "Complaint", "Category", "Urgency"]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "PNR": data.get("pnr") or "",
            "Train": data.get("train") or "",
            "Coach": data.get("coach") or "",
            "Seat": data.get("seat") or "",
            "Complaint": data.get("complaint") or "",
            "Category": category,
            "Urgency": urgency,
        })


def find_officer_and_medical_contacts(train, coach, seat):
    """Return police officer and medical team contact."""
    matched = police_df[
        (police_df["Train Number"] == int(train)) &
        (police_df["Compartment No"] == coach)
    ]
    if matched.empty:
        return None, None, None
    officer_name = matched.iloc[0]["Officer Name"]
    officer_phone = str(matched.iloc[0]["Officer Mobile No."])
    medical_team_phone = str(matched.iloc[0]["Medical Team"])
    return officer_name, officer_phone, medical_team_phone


def process_complaint(data, save_file_path=None):
    """Main processor for complaint handling."""
    complaint_text = data.get("complaint", "")

    # 🔹 Medical emergency
    if is_medical_issue(complaint_text):
        urgency = "High"
        category = "Medical Issue"

        # Get contacts
        officer_name, officer_phone, medical_team_phone = find_officer_and_medical_contacts(
            data.get("train"), data.get("coach"), data.get("seat")
        )

        # Save complaint
        if save_file_path:
            save_complaint(data, category, urgency, filename=save_file_path)

        # Send SMS notifications
        alert_msg = "Medical emergency alert sent."
        if medical_team_phone:
            send_sms(medical_team_phone, f"Medical emergency at Train {data.get('train')} Coach {data.get('coach')} Seat {data.get('seat')}: {complaint_text}")
            alert_msg += f" Medical team notified ({medical_team_phone})."
        if officer_phone:
            send_sms(officer_phone, f"Medical emergency at Train {data.get('train')} Coach {data.get('coach')} Seat {data.get('seat')}: {complaint_text}")
            alert_msg += f" Police notified ({officer_name}, {officer_phone})."

        return category, urgency, alert_msg, officer_name, officer_phone

    # 🔹 Non-medical complaints
    urgency = check_urgency(complaint_text)
    category = classify_issue(complaint_text)

    # Save complaint
    if save_file_path:
        save_complaint(data, category, urgency, filename=save_file_path)

    alert_status = "Complaint logged. Await response."
    officer_name, officer_phone = None, None

    # High urgency non-medical -> alert police
    if urgency.lower() == "high":
        officer_name, officer_phone = police_complaint(
            data.get("train"), data.get("coach"), data.get("seat")
        )
        if officer_phone:
            alert_msg = (
                f"URGENT ALERT\n"
                f"Train: {data.get('train')}\n"
                f"Coach: {data.get('coach')}\n"
                f"Seat: {data.get('seat')}\n"
                f"Complaint: {complaint_text}\n"
                f"Category: {category}\n"
                f"Urgency: {urgency}"
            )
            send_sms(officer_phone, alert_msg)
            alert_status = f"Alert SMS sent to Officer {officer_name} ({officer_phone})."
        else:
            alert_status = "Urgent complaint logged but no police contact found."

    return category, urgency, alert_status, officer_name, officer_phone
