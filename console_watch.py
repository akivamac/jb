import subprocess

MOM = 'ellenfriedman'


def mom_on_console():
    """True if Mom (ellenfriedman) has the physical console session."""
    try:
        out = subprocess.run(['who'], capture_output=True, text=True, timeout=5).stdout or ''
    except Exception:
        return True  # fail-safe: pause on any error
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == MOM and parts[1] == 'console':
            return True
    return False