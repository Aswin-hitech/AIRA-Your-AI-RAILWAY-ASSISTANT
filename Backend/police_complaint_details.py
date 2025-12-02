# police_complaint_details.py
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
CSV_FILE = os.path.join(BASE_DIR, "Storage files", "Police_id.csv")

def format_phone(number: str) -> str:
    """Ensure phone number starts with +91"""
    if not number:
        return ""
    number = number.strip()
    if not number.startswith("+91"):
        number = "+91" + number.lstrip("0").lstrip("+")
    return number

def police_complaint(train, coach, seat=None):
    """
    Returns (officer_name, officer_phone) with +91 formatting.
    """
    with open(CSV_FILE, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["Train Number"].strip() == str(train).strip() and row["Compartment No"].strip() == str(coach).strip():
                officer_name = row.get("Officer Name","").strip()
                officer_phone = format_phone(row.get("Officer Mobile No.",""))
                return officer_name, officer_phone
    return None, None
