from __future__ import annotations

import argparse
import os
from pathlib import Path

from flask import Flask, render_template, request

from runtime_pipeline import SinglePairPtVerifier, decode_upload


RUNTIME_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATCHER_WEIGHTS = RUNTIME_ROOT / "models" / "concrete_matcher.pt"
DEFAULT_YOLO_WEIGHTS = RUNTIME_ROOT / "models" / "concrete_seg_best.pt"


def create_app() -> Flask:
    matcher_weights = Path(os.environ.get("MATCHER_WEIGHTS", str(DEFAULT_MATCHER_WEIGHTS))).resolve()
    yolo_weights = Path(os.environ.get("YOLO_WEIGHTS", str(DEFAULT_YOLO_WEIGHTS))).resolve()
    yolo_device = os.environ.get("YOLO_DEVICE", "cpu")
    threshold = float(os.environ.get("VERIFY_THRESHOLD", "0.5"))

    verifier = SinglePairPtVerifier(
        matcher_weights=matcher_weights,
        yolo_weights=yolo_weights,
        yolo_device=yolo_device,
        imgsz=960,
        conf_thres=0.35,
        warp_canvas_size=1024,
        qr_anchor_size=80,
        qr_anchor_offset=472,
        threshold=threshold,
    )

    app = Flask(__name__, template_folder=str(Path(__file__).with_name("templates")))
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024
    app.config["MATCHER_WEIGHTS"] = str(matcher_weights)
    app.config["YOLO_WEIGHTS"] = str(yolo_weights)
    app.config["YOLO_DEVICE"] = yolo_device
    app.config["VERIFY_THRESHOLD"] = threshold

    def render_page(result=None, error=None):
        return render_template(
            "verify.html",
            result=result,
            error=error,
            matcher_weights=app.config["MATCHER_WEIGHTS"],
            yolo_weights=app.config["YOLO_WEIGHTS"],
            verify_device=app.config["YOLO_DEVICE"],
            verify_threshold=app.config["VERIFY_THRESHOLD"],
        )

    @app.get("/")
    def index():
        return render_page()

    @app.post("/verify")
    def verify():
        left = request.files.get("left_image")
        right = request.files.get("right_image")
        if left is None or right is None or left.filename == "" or right.filename == "":
            return render_page(error="请同时上传两张图片")
        try:
            left_image = decode_upload(left)
            right_image = decode_upload(right)
            return render_page(result=verifier.verify(left_image, right_image))
        except Exception as exc:
            return render_page(error=str(exc))

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动推理网站")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
