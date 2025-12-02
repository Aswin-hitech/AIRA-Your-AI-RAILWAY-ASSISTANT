from getpnr import extract_coach_train_seat_from_railrestro

def prepare_complaint(data):
    """
    Prepares complaint dictionary with proper fields.
    Auto-fills train, coach, seat if PNR is given.
    """
    pnr = data.get("pnr")
    if pnr:
        train, coach, seat = extract_coach_train_seat_from_railrestro(pnr)
        if not train:
            train = data.get("train") or ""
        if not coach:
            coach = data.get("coach") or ""
        if not seat:
            seat = data.get("seat") or ""
    else:
        train = data.get("train") or ""
        coach = data.get("coach") or ""
        seat = data.get("seat") or ""

    complaint = data.get("complaint", "")

    return {
        "pnr": pnr or "",
        "train": train,
        "coach": coach,
        "seat": seat,
        "complaint": complaint
    }
