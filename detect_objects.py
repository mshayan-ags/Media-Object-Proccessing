"""Detect objects in an image, video or webcam with YOLO26 and draw labeled boxes.

Usage:  python detect_objects.py image.jpg              [--conf 0.25] [--model yolo26n.pt]
        python detect_objects.py video.mp4              (saves video_detected.mp4)
        python detect_objects.py 0                      (webcam, press q to quit)
        python detect_objects.py video.mp4 --track      (keep an ID on each object across frames)
        python detect_objects.py image.jpg --classes sphere cube ring   (custom objects)
Setup:  pip install -U ultralytics opencv-python
"""
import argparse
import time
from pathlib import Path

import cv2
from ultralytics import YOLO, YOLOE


PALETTE = [(56, 56, 255), (255, 112, 31), (29, 178, 255), (49, 210, 207), (236, 24, 0),
           (168, 153, 44), (255, 56, 132), (52, 147, 26), (187, 212, 0), (133, 0, 82)]  # BGR
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def draw_box(img, xyxy, label, color):
    x1, y1, x2, y2 = map(int, xyxy)
    thick = max(2, round(sum(img.shape[:2]) / 600))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)

    font, scale = cv2.FONT_HERSHEY_SIMPLEX, thick / 3
    (tw, th), base = cv2.getTextSize(label, font, scale, max(1, thick // 2))
    ty = y1 - 4 if y1 - th - base - 4 > 0 else y1 + th + base + 4  # keep label inside image
    cv2.rectangle(img, (x1, ty - th - base), (x1 + tw + 4, ty + base // 2), color, -1)
    cv2.putText(img, label, (x1 + 2, ty - base // 2), font, scale, (255, 255, 255),
                max(1, thick // 2), cv2.LINE_AA)


def annotate(img, result, verbose=False):
    """Draw every detection in `result` onto `img`; returns number of objects."""
    ids = result.boxes.id  # only set when tracking
    for i, box in enumerate(result.boxes):
        cls, conf = int(box.cls), float(box.conf)
        name = result.names[cls]
        tag = f"#{int(ids[i])} " if ids is not None else ""
        draw_box(img, box.xyxy[0].tolist(), f"{tag}{name} {conf:.2f}", PALETTE[cls % len(PALETTE)])
        if verbose:
            print(f"  {tag}{name:<15} {conf:.2f}  box={[round(v) for v in box.xyxy[0].tolist()]}")
    return len(result.boxes)


def run_image(model, args):
    result = model.predict(args.source, conf=args.conf)[0]
    img = result.orig_img.copy()
    print(f"Found {len(result.boxes)} objects:")
    annotate(img, result, verbose=True)

    out = Path(args.source).with_name(Path(args.source).stem + "_detected.jpg")
    cv2.imwrite(str(out), img)
    print(f"Saved: {out}")
    if args.show:
        cv2.imshow("YOLO26 detections (press any key)", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(model, args):
    webcam = args.source.isdigit()
    cap = cv2.VideoCapture(int(args.source) if webcam else args.source)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video source: {args.source}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not webcam else 0
    out = Path("webcam_detected.mp4" if webcam else
               Path(args.source).with_name(Path(args.source).stem + "_detected.mp4"))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    n, live_fps = 0, 0.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.time()
        if args.track:
            result = model.track(frame, conf=args.conf, persist=True, verbose=False)[0]
        else:
            result = model.predict(frame, conf=args.conf, verbose=False)[0]
        count = annotate(frame, result)
        n += 1

        inst = 1 / max(time.time() - t0, 1e-6)
        live_fps = inst if n == 1 else 0.9 * live_fps + 0.1 * inst  # smoothed
        cv2.rectangle(frame, (0, h - 34), (260, h), (0, 0, 0), -1)  # status bar, bottom-left
        cv2.putText(frame, f"{count} objects | {live_fps:.1f} FPS", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        writer.write(frame)

        if total:
            print(f"\rFrame {n}/{total}  ({live_fps:.1f} FPS)", end="", flush=True)
        if args.show:
            cv2.imshow("YOLO26 detections (press q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"\nProcessed {n} frames. Saved: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="image path, video path, or webcam index (e.g. 0)")
    ap.add_argument("--model", default="yolo26n.pt", help="yolo26n/s/m/l/x.pt (auto-downloads)")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--classes", nargs="+", help="custom objects, e.g. --classes sphere cube ring")
    ap.add_argument("--track", action="store_true", help="video: keep an ID on each object")
    ap.add_argument("--no-show", dest="show", action="store_false", help="don't open a window")
    args = ap.parse_args()

    if args.classes:  # open-vocabulary: detect anything you can name
        model = YOLOE("yoloe-26s-seg.pt")
        model.set_classes(args.classes, model.get_text_pe(args.classes))
    else:             # standard 80 COCO classes (person, car, dog, ...)
        model = YOLO(args.model)

    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(model, args)
    else:
        run_video(model, args)


if __name__ == "__main__":
    main()
