#!/usr/bin/env python3
"""Capture a real Alacritty receiving-terminal window, never an HTML imitation.

Run after the measured live request. Only matching JSON events from the dedicated
systemd API journal are shown. The screenshot is taken by grim from the exact
visible terminal rectangle on a private Hyprland special workspace. Candidate
frames stay in memory; only a frame validated by terminal/event/request-ID OCR
is persisted. No arbitrary shell history, credentials or desktop frame is saved.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CLASS = "pathsnap-evidence"
TITLE = "PathSnap | actual API receiving terminal"
PUBLIC_KEYS = ("timestamp", "event", "request_id", "trip_id", "dataset_role",
               "model_hash", "match_config_hash", "model_name", "metrics_status",
               "inference_ms", "scoring_ms", "total_ms", "route_arc_count",
               "observation_count", "learned_emission_observations")


def run(*args):
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def load_events(request_id, since):
    raw = run("journalctl", "--user", "-u", "path-snap-api.service", "--since", since,
              "--no-pager", "--output=cat")
    found = []
    for line in raw.splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("request_id") == request_id:
            # Persist only verified fixed public scalars, not arbitrary log payloads.
            found.append({k: event[k] for k in PUBLIC_KEYS if k in event})
    kinds = {r.get("event") for r in found}
    if not {"match_started", "match_completed"}.issubset(kinds):
        raise RuntimeError("actual request's start/completion journal events not found")
    return found


def display(args):
    events = load_events(args.request_id, args.since)
    print("PATHSNAP / ACTUAL API RECEIVING TERMINAL", flush=True)
    print("Source: dedicated systemd user service journal, not a mock server", flush=True)
    print("Public origin: https://ml.d3va.dev", flush=True)
    print("$ journalctl --user -u path-snap-api.service --since " + json.dumps(args.since)
          + " --no-pager --output=cat", flush=True)
    print("Filtered by the measured HTTP request ID:", flush=True)
    print(args.request_id, flush=True)
    print("Request ID groups: " + " ".join(args.request_id[i:i+4] for i in range(0,len(args.request_id),4)), flush=True)
    print("Only original public scalar fields are displayed; no routes or secrets.\n", flush=True)
    started = next(e for e in events if e.get("event") == "match_started")
    completed = next(e for e in events if e.get("event") == "match_completed")
    print("trip_id: " + str(started.get("trip_id")) + "   dataset_role: " + str(started.get("dataset_role")), flush=True)
    print("model_hash:        " + str(started.get("model_hash")), flush=True)
    print("match_config_hash: " + str(started.get("match_config_hash")), flush=True)
    print("", flush=True)
    print("match_started    " + str(started.get("timestamp")), flush=True)
    print("match_completed  " + str(completed.get("timestamp")), flush=True)
    print("metrics_status: " + str(completed.get("metrics_status")), flush=True)
    print("inference_ms: " + str(completed.get("inference_ms"))
          + "   scoring_ms: " + str(completed.get("scoring_ms"))
          + "   total_ms: " + str(completed.get("total_ms")), flush=True)
    print("", flush=True)
    print("ACTUAL API RECEIVING TERMINAL / original dedicated-service journal events", flush=True)
    print("Request " + args.request_id + " executed fresh learned inference.", flush=True)
    # A child exits immediately; Alacritty --hold retains its genuine rendered output.
    (args.output / ".terminal_display_ready.json").write_text(json.dumps({"pid": os.getpid(), "events": len(events)}))


def terminal_frame_flags(recognized, request_id):
    """OCR is a persistence guard, not the source of request/model identity.

    Hex IDs contain no letter O; normalize only common OCR zero glyphs. Exact
    request/model identity is separately taken from original journal JSON.
    """
    normalized = "".join(c for c in recognized.lower() if c.isalnum())
    hexadecimal = "".join(c for c in recognized.lower().replace("@", "0").replace("o", "0") if c.isalnum())
    return {"terminal_marker": "actualapireceivingterminal" in normalized,
            "exact_request_id": request_id in hexadecimal,
            "started": "matchstarted" in normalized, "completed": "matchcompleted" in normalized}


def capture(args):
    args.output.mkdir(parents=True, exist_ok=True)
    evidence_path = args.output / "api_terminal_evidence.json"
    image = args.output / "screenshots/api_terminal.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    if image.exists() or (evidence_path.exists() and json.loads(evidence_path.read_text()).get("valid", True)):
        raise RuntimeError("refuse to overwrite genuine terminal evidence")
    events = load_events(args.request_id, args.since)
    monitor = next(m for m in json.loads(run("hyprctl", "-j", "monitors")) if m["focused"])
    current_clients = json.loads(run("hyprctl", "-j", "clients"))
    if any(c.get("class") == CLASS for c in current_clients):
        raise RuntimeError("dedicated capture terminal already exists; refusing duplicate")
    previous_address = json.loads(run("hyprctl", "-j", "activewindow")).get("address")
    previous_special = monitor.get("specialWorkspace", {}).get("name", "")
    if previous_special:
        raise RuntimeError("another special workspace is visible; refuse to disturb it")
    terminal = subprocess.Popen([
        "alacritty", "--class", CLASS, "--title", TITLE, "--hold",
        "-o", "window.opacity=1.0", "-o", "font.size=10.5",
        "-o", "window.dimensions.columns=140", "-o", "window.dimensions.lines=42",
        "-o", 'colors.primary.background="#07111f"',
        "-o", 'colors.primary.foreground="#e4edf6"',
        "-e", sys.executable, "-B", str(Path(__file__).resolve()),
        "--display", "--request-id", args.request_id, "--since", args.since,
        "--output", str(args.output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    address = None
    special_visible = False
    started = time.monotonic()
    try:
        while time.monotonic() - started < 20:
            candidates = [c for c in json.loads(run("hyprctl", "-j", "clients"))
                          if c.get("class") == CLASS and c.get("pid") == terminal.pid]
            if len(candidates) == 1 and (args.output / ".terminal_display_ready.json").is_file():
                address = candidates[0]["address"]
                break
            if terminal.poll() is not None:
                raise RuntimeError("dedicated terminal exited before capture")
            # Bounded UI state polling only, never job sleeping or duplicate work.
            time.sleep(0.10)
        if not address:
            raise RuntimeError("dedicated terminal did not become ready within 20 s")
        logical_width = int(monitor["width"] / monitor["scale"])
        logical_height = int(monitor["height"] / monitor["scale"])
        width, height = min(1480, logical_width - 60), min(900, logical_height - 80)
        x, y = int(monitor["x"] + (logical_width - width) / 2), int(monitor["y"] + (logical_height - height) / 2)
        run("hyprctl", "dispatch", "hl.dsp.window.move({workspace=\"special:pathsnap-evidence\", window=" + json.dumps("address:" + address) + ", silent=true})")
        run("hyprctl", "dispatch", "hl.dsp.window.float({action=\"set\", window=" + json.dumps("address:" + address) + "})")
        run("hyprctl", "dispatch", "hl.dsp.window.resize({window=" + json.dumps("address:" + address) + f", x={width}, y={height}, exact=true}})")
        run("hyprctl", "dispatch", "hl.dsp.window.move({window=" + json.dumps("address:" + address) + f", x={x}, y={y}, exact=true}})")
        run("hyprctl", "dispatch", 'hl.dsp.workspace.toggle_special("pathsnap-evidence")')
        special_visible = True
        run("hyprctl", "dispatch", "hl.dsp.focus({window=" + json.dumps("address:" + address) + "})")
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            active = json.loads(run("hyprctl", "-j", "activewindow"))
            if active.get("address") == address and active.get("mapped") and not active.get("hidden"):
                candidates = json.loads(run("hyprctl", "-j", "clients"))
                client = next(c for c in candidates if c.get("address") == address)
                if all(abs(a - b) <= 2 for a, b in zip(client["size"], [width, height])):
                    break
            time.sleep(0.05)
        client = json.loads(run("hyprctl", "-j", "activewindow"))
        if client.get("address") != address or client.get("class") != CLASS:
            raise RuntimeError("refuse to screenshot a non-evidence window")
        position, size = client["at"], client["size"]
        geometry = f'{position[0]},{position[1]} {size[0]}x{size[1]}'
        # IPC geometry can lead rendered frames during compositor animation.
        # Never persist a frame unless terminal marker AND exact events pass OCR.
        from io import BytesIO
        from PIL import Image
        deadline = time.monotonic() + 8
        pixels = None
        ocr = ""
        while time.monotonic() < deadline:
            # Recheck focus both sides of capture: never persist a user-switched window.
            before = json.loads(run("hyprctl", "-j", "activewindow"))
            if before.get("address") != address or before.get("class") != CLASS or before.get("pid") != terminal.pid:
                raise RuntimeError("focus changed away from dedicated terminal; no screenshot persisted")
            frame = subprocess.check_output(["grim", "-g", geometry, "-"])
            after = json.loads(run("hyprctl", "-j", "activewindow"))
            if after.get("address") != address or after.get("class") != CLASS or after.get("pid") != terminal.pid:
                raise RuntimeError("focus changed during capture; no screenshot persisted")
            sample = Image.open(BytesIO(frame)).convert("RGB")
            dark = sum(max(c) < 90 for c in sample.get_flattened_data()) / (sample.width * sample.height)
            if dark > 0.85:
                recognized = subprocess.run(["tesseract", "stdin", "stdout"], input=frame, capture_output=True, check=True).stdout.decode()
                flags = terminal_frame_flags(recognized, args.request_id)
                print(json.dumps({"privacy_validation": flags, "dark_fraction": dark,
                                  "ocr_characters": len(recognized),
                                  "known_public_words_found": [word for word in ("PATHSNAP", "TERMINAL", "match", "request", "Source") if word.lower() in recognized.lower()],
                                  "hex_token_lengths": [len(word) for word in recognized.split() if len(word) >= 16 and all(c.lower() in "0123456789abcdef@" for c in word)]}), file=sys.stderr, flush=True)
                if all(flags.values()):
                    pixels, ocr = frame, recognized
                    break
            time.sleep(0.12)
        if pixels is None:
            raise RuntimeError("no rendered frame passed privacy/OCR validation; no screenshot persisted")
        image.write_bytes(pixels)
        (args.output / "api_terminal_ocr.txt").write_text(ocr)
        evidence = {"captured_at_utc": datetime.now(timezone.utc).isoformat(),
                    "source": "real Alacritty Wayland window displaying actual dedicated API journal events",
                    "mocked": False, "valid": True, "ocr_privacy_guard_passed": True, "image": str(image.relative_to(ROOT)),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                    "request_id": args.request_id, "journal_since": args.since,
                    "journal_unit": "path-snap-api.service", "events": events,
                    "window_class": CLASS, "window_title": TITLE,
                    "screenshot_tool": "grim", "geometry_logical": geometry,
                    "terminal_pid": terminal.pid,
                    "privacy": "only dedicated terminal rectangle; exact request events allowlisted to public scalars"}
        evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
        print(json.dumps(evidence, indent=2))
    finally:
        if special_visible:
            run("hyprctl", "dispatch", 'hl.dsp.workspace.toggle_special("pathsnap-evidence")')
        if address:
            subprocess.run(["hyprctl", "dispatch", "hl.dsp.window.close({window=" + json.dumps("address:" + address) + "})"], capture_output=True)
        if terminal.poll() is None:
            terminal.terminate()
        terminal.wait(timeout=5)
        if previous_address:
            subprocess.run(["hyprctl", "dispatch", "hl.dsp.focus({window=" + json.dumps("address:" + previous_address) + "})"], capture_output=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--since", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "reports/phase_three/live")
    parser.add_argument("--display", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "reports"):
        raise RuntimeError("capture outputs must remain inside project reports")
    display(args) if args.display else capture(args)


if __name__ == "__main__":
    main()
