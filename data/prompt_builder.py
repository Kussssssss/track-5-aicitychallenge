PHASE_MAP = {
    "0": "pre-recognition",
    "1": "recognition",
    "2": "judgment",
    "3": "action",
    "4": "avoidance",
}


def normalize_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(v) for v in value if v is not None).strip()
    return str(value).strip()


def phase_label_from_id(phase_id):
    if phase_id is None:
        return ""
    return PHASE_MAP.get(str(phase_id), str(phase_id))


def build_prompt(caption_pedestrian="", caption_vehicle="", phase_label=""):
    parts = ["Fixed traffic camera future prediction."]
    if phase_label:
        parts.append(f"Phase: {phase_label}.")
    if caption_pedestrian:
        parts.append(f"Pedestrian behavior: {caption_pedestrian}")
    if caption_vehicle:
        parts.append(f"Vehicle behavior: {caption_vehicle}")
    parts.append(
        "Preserve the same camera view, road layout, lighting, vehicle identity, "
        "pedestrian location, and realistic traffic motion."
    )
    parts.append("Generate realistic future traffic frames.")
    return " ".join(parts)

