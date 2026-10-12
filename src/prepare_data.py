"""Download MS-COCO 2017 val, build the prompt splits and the FID reference images.

    python -m src.prepare_data

Outputs (all under data/, which is git-ignored):
    splits/tune.jsonl       500 image_ids, for parameter search
    splits/eval.jsonl       2,000 image_ids, for the final comparison (disjoint from tune)
    splits/diversity.jsonl  100 prompts drawn from eval, for the LPIPS diversity study
    coco_ref/<split>/       real images, centre-cropped and resized to 512x512, for FID
"""

import argparse
import json
import random
import urllib.request
import zipfile
from collections import defaultdict

from PIL import Image

from src.common import DATA_DIR, SPLITS_DIR

COCO_DIR = DATA_DIR / "coco"
REF_DIR = DATA_DIR / "coco_ref"
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/zips/val2017.zip"

SPLIT_SEED = 2026
N_TUNE, N_EVAL, N_DIVERSITY = 500, 2000, 100
REF_SIZE = 512


def download(url, dest):
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}", flush=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)


def extract(zip_path, member_prefix, target):
    """Extract once; a marker file written at the end makes an interrupted extraction rerun."""
    marker = target.with_name(target.name + ".extracted")
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(COCO_DIR, [m for m in zf.namelist() if m.startswith(member_prefix)])
    marker.touch()


def first_captions():
    """image_id -> (file_name, caption with the lowest annotation id)."""
    with open(COCO_DIR / "annotations" / "captions_val2017.json", encoding="utf-8") as f:
        data = json.load(f)
    files = {img["id"]: img["file_name"] for img in data["images"]}
    caps = defaultdict(list)
    for ann in data["annotations"]:
        caps[ann["image_id"]].append((ann["id"], " ".join(ann["caption"].split())))
    return {iid: (files[iid], min(c)[1]) for iid, c in caps.items()}


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_reference(rows, split):
    out = REF_DIR / split
    out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        dest = out / f"{row['prompt_id']}.png"
        if dest.exists():
            continue
        img = Image.open(COCO_DIR / "val2017" / row["file_name"]).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side)).resize((REF_SIZE, REF_SIZE), Image.BICUBIC)
        tmp = dest.with_name(dest.name + ".tmp")
        img.save(tmp, format="PNG")
        tmp.replace(dest)  # never leave a truncated reference image behind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-images", action="store_true", help="only build prompt splits")
    args = ap.parse_args()

    download(ANN_URL, COCO_DIR / "annotations_trainval2017.zip")
    extract(COCO_DIR / "annotations_trainval2017.zip", "annotations/captions_", COCO_DIR / "annotations")

    captions = first_captions()
    ids = sorted(captions)
    random.Random(SPLIT_SEED).shuffle(ids)
    to_row = lambda iid: {
        "prompt_id": f"coco{iid}", "image_id": iid,
        "file_name": captions[iid][0], "prompt": captions[iid][1],
    }
    tune = [to_row(i) for i in ids[:N_TUNE]]
    eval_ = [to_row(i) for i in ids[N_TUNE : N_TUNE + N_EVAL]]
    diversity = random.Random(SPLIT_SEED + 1).sample(eval_, N_DIVERSITY)

    write_jsonl(SPLITS_DIR / "tune.jsonl", tune)
    write_jsonl(SPLITS_DIR / "eval.jsonl", eval_)
    write_jsonl(SPLITS_DIR / "diversity.jsonl", diversity)
    with open(SPLITS_DIR / "manifest.json", "w", encoding="utf-8") as f:
        json.dump({"seed": SPLIT_SEED, "tune": len(tune), "eval": len(eval_),
                   "diversity": len(diversity), "caption_rule": "lowest annotation id"}, f, indent=2)
    print(f"splits: tune={len(tune)} eval={len(eval_)} diversity={len(diversity)}", flush=True)

    if not args.skip_images:
        download(IMG_URL, COCO_DIR / "val2017.zip")
        extract(COCO_DIR / "val2017.zip", "val2017/", COCO_DIR / "val2017")
        make_reference(tune, "tune")
        make_reference(eval_, "eval")
        print("reference images ready", flush=True)


if __name__ == "__main__":
    main()
