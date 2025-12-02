# medical_finder.py
import re
import pandas as pd

MEDICAL_KEYWORDS = [
    "emergency","medical emergency","first aid","injury","accident","trauma","bleeding",
    "fracture","burn","shock","breathing difficulty","chest pain","heart attack",
    "stroke","unconscious","fainting","seizure","allergic reaction","asthma attack",
    "diabetes emergency","hypoglycemia","hyperglycemia","fever","headache","nausea",
    "vomiting","dizziness","dehydration","heat stroke","hypothermia","infection",
    "poisoning","drug overdose","pregnancy emergency","labor pain","childbirth",
    "choking","airway obstruction","respiratory distress","cardiac arrest","CPR",
    "AED","ambulance","medical kit","sick passenger","illness","paramedics","nurse",
    "doctor","hospital","emergency care","pulse","blood pressure","oxygen saturation",
    "emergency medication","wound","bandage","splint","tourniquet","ICU","oxygen",
    "defibrillator","heartattack","heart"
]

MEDICAL_PATTERNS = [re.compile(rf"\b{kw}\b", re.IGNORECASE) for kw in MEDICAL_KEYWORDS]

def format_phone(number: str) -> str:
    """Ensure phone number starts with +91"""
    if not number:
        return ""
    number = number.strip()
    if not number.startswith("+91"):
        number = "+91" + number.lstrip("0").lstrip("+")
    return number

def is_medical_issue(complaint_text: str) -> bool:
    if not complaint_text:
        return False
    text = complaint_text.lower()
    for pat in MEDICAL_PATTERNS:
        if pat.search(text):
            return True
    return False

def medical_complaint(train, coach, seat, complaint_text, urgency, police_df):
    if urgency.lower() != "high":
        return None, None
    matched = police_df[
        (police_df["Train Number"] == int(train)) &
        (police_df["Compartment No"] == coach)
    ]
    if matched.empty:
        return None, None
    medical_team_number = format_phone(str(matched.iloc[0]["Medical Team"]))
    medical_team_name = f"Medical Team - {coach}"
    return medical_team_name, medical_team_number
