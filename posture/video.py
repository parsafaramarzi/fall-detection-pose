"""Run the fall rule on a video with a YOLO pose model.

Needs the optional dependencies in requirements-video.txt. The model weights
(`yolo11x-pose.pt` by default) are downloaded by ultralytics on first use.
"""
from . import fall
from .visibility import visibility_score

COLORS = {fall.STEADY: (0, 200, 0), fall.FALLEN: (0, 0, 255), fall.UNKNOWN: (128, 128, 128)}  # BGR


def analyse_frame(result):
    """Per-person list of (box_xyxy, status, detail) for one ultralytics result."""
    if result.boxes.xyxy.numel() == 0:
        return []
    out = []
    for box, kp, cf in zip(result.boxes.xyxy, result.keypoints.xy, result.keypoints.conf):
        kp, cf = kp.cpu().numpy(), cf.cpu().numpy()
        p = fall.classify_posture(kp, cf)
        out.append((box.cpu().numpy(), p.status, f"{p.reason}; visible {visibility_score(cf):.0%}"))
    return out


def run(video_path, output_path, weights="yolo11x-pose.pt", width=960):
    """Annotate a video; returns the number of person-frames flagged as fallen."""
    import cv2
    import imageio
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    writer = imageio.get_writer(str(output_path), fps=cap.get(cv2.CAP_PROP_FPS) or 30,
                                codec="libx264", quality=8)
    n_fallen = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(width * h / w)))
        for box, status, detail in analyse_frame(model(frame, classes=[0], verbose=False)[0]):
            x1, y1, x2, y2 = map(int, box)
            color = COLORS[status]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, status, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            n_fallen += status == fall.FALLEN
        writer.append_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    writer.close()
    return n_fallen
