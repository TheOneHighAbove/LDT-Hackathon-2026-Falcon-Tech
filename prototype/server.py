"""Serve a dependency-light explorer for frozen speed-release artifacts."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

from PIL import Image


def _read_annotations(path: Path, split: str) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"image_id", "x", "y", "w", "h"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"invalid annotation schema: {path}")
    for row in rows:
        row["split"] = split
        for name in ("x", "y", "w", "h"):
            row[name] = int(float(row[name]))
    return rows


@dataclass(frozen=True)
class ExplorerData:
    images_dir: Path
    query_ids: tuple[str, ...]
    annotations: dict[str, dict]
    ranking: dict[str, tuple[str, ...]]
    accepted: dict[str, dict]
    manifest: dict

    @classmethod
    def load(cls, dataset_dir: Path, output_dir: Path) -> ExplorerData:
        query = _read_annotations(dataset_dir / "test_query.csv", "query")
        gallery = _read_annotations(dataset_dir / "test_gallery.csv", "gallery")
        annotations = {str(row["image_id"]): row for row in [*query, *gallery]}
        query_ids = tuple(str(row["image_id"]) for row in query)
        gallery_ids = {str(row["image_id"]) for row in gallery}
        if len(annotations) != len(query) + len(gallery):
            raise ValueError("query and gallery image_id values must be unique")

        ranking: dict[str, tuple[str, ...]] = {}
        with (output_dir / "submission.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            for number, row in enumerate(csv.reader(stream), start=1):
                if len(row) != 11:
                    raise ValueError(f"submission row {number} must contain 11 columns")
                query_id, gallery_top10 = row[0], tuple(row[1:])
                if query_id in ranking:
                    raise ValueError(f"duplicate submission query_id: {query_id}")
                if len(set(gallery_top10)) != 10:
                    raise ValueError(
                        f"submission query {query_id} has duplicate gallery IDs"
                    )
                if not set(gallery_top10).issubset(gallery_ids):
                    raise ValueError(
                        f"submission query {query_id} references unknown gallery ID"
                    )
                ranking[query_id] = gallery_top10
        if tuple(ranking) != query_ids:
            raise ValueError("submission query order does not match test_query.csv")

        accepted: dict[str, dict] = {}
        with (output_dir / "candidates.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as stream:
            for row in csv.DictReader(stream):
                query_id = str(row["query_id"])
                gallery_id = str(row["gallery_id"])
                confidence = float(row["confidence"])
                if query_id in accepted:
                    raise ValueError(f"duplicate candidate query_id: {query_id}")
                if query_id not in ranking:
                    raise ValueError(f"candidate has unknown query_id: {query_id}")
                if gallery_id != ranking[query_id][0]:
                    raise ValueError(
                        f"candidate for {query_id} does not match submission top-1"
                    )
                if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                    raise ValueError(f"invalid confidence for query {query_id}")
                accepted[query_id] = {
                    "gallery_id": gallery_id,
                    "confidence": confidence,
                }
        manifest = json.loads(
            (output_dir / "inference_manifest.json").read_text(encoding="utf-8")
        )
        refusal = manifest.get("refusal", {})
        if int(refusal.get("accepted", len(accepted))) != len(accepted):
            raise ValueError("manifest accepted count does not match candidates.csv")
        if int(refusal.get("refused", len(query_ids) - len(accepted))) != (
            len(query_ids) - len(accepted)
        ):
            raise ValueError("manifest refused count does not match candidates.csv")
        return cls(
            images_dir=dataset_dir / "images",
            query_ids=query_ids,
            annotations=annotations,
            ranking=ranking,
            accepted=accepted,
            manifest=manifest,
        )

    def result(self, query_id: str) -> dict:
        if query_id not in self.ranking:
            raise KeyError(query_id)
        candidate = self.accepted.get(query_id)
        return {
            "query_id": query_id,
            "accepted": candidate is not None,
            "confidence": None if candidate is None else candidate["confidence"],
            "candidate_id": None if candidate is None else candidate["gallery_id"],
            "query_image": f"/image/{query_id}",
            "top10": [
                {
                    "rank": rank,
                    "gallery_id": gallery_id,
                    "image": f"/image/{gallery_id}",
                }
                for rank, gallery_id in enumerate(self.ranking[query_id], start=1)
            ],
        }

    def cropped_jpeg(self, image_id: str) -> bytes:
        row = self.annotations.get(image_id)
        if row is None:
            raise KeyError(image_id)
        return _cropped_jpeg(
            str(self.images_dir / f"{image_id}.jpg"),
            row["x"],
            row["y"],
            row["w"],
            row["h"],
        )

    def health(self) -> dict:
        refusal = self.manifest.get("refusal", {})
        return {
            "status": "ok",
            "profile": self.manifest.get("profile", "speed"),
            "queries": len(self.query_ids),
            "accepted": int(refusal.get("accepted", len(self.accepted))),
            "refused": int(
                refusal.get("refused", len(self.query_ids) - len(self.accepted))
            ),
            "artifact_mode": "frozen_batch_results",
        }


@lru_cache(maxsize=512)
def _cropped_jpeg(path: str, x: int, y: int, w: int, h: int) -> bytes:
    with Image.open(path) as source:
        image = source.convert("RGB")
        pad_x = round(max(0, w) * 0.02)
        pad_y = round(max(0, h) * 0.02)
        left = max(0, x - pad_x)
        top = max(0, y - pad_y)
        right = min(image.width, x + w + pad_x)
        bottom = min(image.height, y + h + pad_y)
        image = image.crop((left, top, right, bottom))
        image.thumbnail((720, 540), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=88, optimize=True)
    return output.getvalue()


def _handler(data: ExplorerData, html: bytes):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, payload: bytes, content_type: str, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _json(self, value: object, status=HTTPStatus.OK):
            self._send(
                json.dumps(value, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/":
                self._send(html, "text/html; charset=utf-8")
                return
            if path == "/api/health":
                self._json(data.health())
                return
            if path == "/api/queries":
                self._json(
                    [
                        {"query_id": query_id, "accepted": query_id in data.accepted}
                        for query_id in data.query_ids
                    ]
                )
                return
            if path.startswith("/api/result/"):
                query_id = unquote(path.removeprefix("/api/result/"))
                try:
                    self._json(data.result(query_id))
                except KeyError:
                    self._json({"error": "unknown query_id"}, HTTPStatus.NOT_FOUND)
                return
            if path.startswith("/image/"):
                image_id = unquote(path.removeprefix("/image/"))
                try:
                    payload = data.cropped_jpeg(image_id)
                except KeyError:
                    self._json({"error": "unknown image_id"}, HTTPStatus.NOT_FOUND)
                    return
                self._send(payload, "image/jpeg")
                return
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

        def log_message(self, format, *args):
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/score_optimized_speed"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and print one result without starting a server",
    )
    args = parser.parse_args()
    data = ExplorerData.load(args.dataset_dir, args.output_dir)
    if args.check:
        print(json.dumps(data.health(), indent=2), flush=True)
        print(json.dumps(data.result(data.query_ids[0]), indent=2), flush=True)
        return
    html = (Path(__file__).parent / "index.html").read_bytes()
    server = ThreadingHTTPServer((args.host, args.port), _handler(data, html))
    print(f"Result explorer: http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
