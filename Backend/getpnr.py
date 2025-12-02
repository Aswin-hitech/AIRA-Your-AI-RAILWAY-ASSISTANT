import requests
from bs4 import BeautifulSoup
import re

def extract_coach_train_seat_from_railrestro(pnr):
    url = f"https://www.railrestro.com/check-pnr-status?pnr={pnr}#appRatingModal"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    try:
        res = requests.get(url, headers=headers, timeout=12)
        res.raise_for_status()
    except Exception as e:
        print("Request error:", e)
        return None, None, None

    soup = BeautifulSoup(res.text, "html.parser")

    # --- Extract Train Number ---
    train_number = None
    for a in soup.find_all("a"):
        text = a.get_text(strip=True)
        match = re.search(r'EXPRESS\s*(\d{5})', text)
        if match:
            train_number = match.group(1)
            break
    if not train_number:
        all_text = soup.get_text(separator=" ")
        match = re.search(r'EXPRESS\s*(\d{5})', all_text)
        if match:
            train_number = match.group(1)

    # --- Extract Coach and Seat ---
    coach, seat = None, None
    found = False
    for td in soup.find_all(['td', 'span', 'div']):
        val = td.get_text(strip=True)
        m = re.search(r"CNF\s*([A-Z]+\d+)\s*(\d+)?", val)
        if m:
            coach = m.group(1)
            seat = m.group(2)
            found = True
            break
    if not found:
        for td in soup.find_all(['td', 'span', 'div']):
            val = td.get_text(strip=True)
            parts = val.split()
            if len(parts) == 2 and re.match(r"^[A-Z]{1,2}\d{1,2}$", parts[0], re.I):
                coach = parts[0]
                seat = parts[1]
                break

    return train_number, coach, seat
