from datetime import datetime

def get_current_time() -> str:
    """Returns the current system time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

