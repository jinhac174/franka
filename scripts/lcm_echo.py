"""See what is flowing between the processes (a text-only lcm-spy).

    python scripts/lcm_echo.py                          # every channel: type and rate
    python scripts/lcm_echo.py franka_arm_joint_meas    # also print the newest message

Run it next to a running sim (or robot). The message type is found from the
8-byte fingerprint every LCM message starts with, matched against the classes
in ``lcm_type/``.
"""

import importlib
import os
import pkgutil
import sys
import time

import lcm
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lcm_type  # noqa: E402


def message_types() -> dict:
    """fingerprint bytes -> LCM message class, for every class in lcm_type/."""
    types = {}
    for mod in pkgutil.walk_packages(lcm_type.__path__, "lcm_type."):
        if mod.ispkg:
            continue
        m = importlib.import_module(mod.name)
        cls = getattr(m, mod.name.rsplit(".", 1)[1], None)
        if cls is not None and hasattr(cls, "_get_packed_fingerprint"):
            types[cls._get_packed_fingerprint()] = cls
    return types


def show(msg) -> str:
    lines = []
    for f in msg.__slots__:
        v = getattr(msg, f)
        if isinstance(v, (list, tuple)) and v and isinstance(v[0], float):
            v = np.array2string(np.asarray(v), precision=3, suppress_small=True, max_line_width=100)
        elif isinstance(v, (list, tuple)) and len(v) > 12:
            v = f"[{len(v)} items]"
        lines.append(f"  {f:12s} {v}")
    return "\n".join(lines)


def main():
    watch = set(sys.argv[1:])
    types = message_types()
    stats, newest = {}, {}

    def on_msg(channel, data):
        cls = types.get(data[:8])
        stats.setdefault(channel, [0, cls.__name__ if cls else "?"])[0] += 1
        if channel in watch and cls is not None:
            newest[channel] = cls.decode(data)

    lc = lcm.LCM()
    lc.subscribe(".*", on_msg)
    print(f"Listening on {os.environ.get('LCM_DEFAULT_URL', 'the default LCM URL')} (Ctrl-C to stop)")
    t0 = time.time()
    try:
        while True:
            try:
                lc.handle_timeout(50)
            except OSError:  # Ctrl-C arrived while LCM was waiting
                break
            dt = time.time() - t0
            if dt < 1.0:
                continue
            print(f"\n--- {time.strftime('%H:%M:%S')} ---")
            if not stats:
                print("  nothing received: is a sim / robot process running?")
            for ch in sorted(stats):
                n, name = stats[ch]
                print(f"  {ch:32s} {name:18s} {n / dt:8.1f} msg/s")
            for ch in sorted(newest):
                print(f"\n{ch}:\n{show(newest[ch])}")
            stats.clear()
            newest.clear()
            t0 = time.time()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
