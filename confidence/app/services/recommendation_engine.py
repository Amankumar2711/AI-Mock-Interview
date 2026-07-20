def generate_recommendation(overall):
    if overall >= 80:
        return {
            "status": "Proceed",
            "remarks": "Strong communication"
        }
    elif overall >= 60:
        return {
            "status": "Borderline",
            "remarks": "Needs improvement"
        }
    return {
        "status": "Reject",
        "remarks": "Communication below threshold"
    }
