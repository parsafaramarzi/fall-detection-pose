"""Annotate a video:  python run_video.py input.mp4 output.mp4 --mode fall|movement"""
import argparse

from posture.video import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--mode", choices=["fall", "movement"], default="fall")
    ap.add_argument("--weights", default="yolo11x-pose.pt")
    a = ap.parse_args()
    n = run(a.input, a.output, a.mode, a.weights)
    print(f"done; {n} person-frames flagged as fallen" if a.mode == "fall" else "done")
