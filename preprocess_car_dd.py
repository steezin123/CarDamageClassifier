import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
from PIL import Image, ImageDraw

#Torch for tensors and normalization
try:
    import torch
    from torch.utils.data import Dataset
except ImportError:
    torch = None

# Helpers
def load_coco(ann_path: str) -> Dict[str, Any]:
    with open(ann_path, "r") as f:
        coco = json.load(f)
    # Minimal validation
    assert "images" in coco and "annotations" in coco, "COCO JSON must have 'images' and 'annotations'."
    return coco

def build_indices(coco: Dict[str, Any]) -> Tuple[Dict[int, Dict], Dict[int, List[Dict]], Dict[int, str]]:
    images_by_id = {img["id"]: img for img in coco.get("images", [])}
    anns_by_img = {}
    for ann in coco.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    # Categories 
    cat_name = {}
    for c in coco.get("categories", []):
        cid = c["id"]
        cat_name[cid] = c.get("name", str(cid))
    return images_by_id, anns_by_img, cat_name

def resize_and_letterbox(img: Image.Image, target_size: Optional[int]) -> Tuple[Image.Image, float, Tuple[int,int]]:
    """
    Keeps aspect ratio, fits longest side to target_size, pads to square (target_size).
    Returns resized+letterboxed image, scaling factor, and padding (left, top).
    If target_size is None, returns original.
    """
    if target_size is None:
        return img, 1.0, (0, 0)
    w, h = img.size
    if w == h and w == target_size:
        return img, 1.0, (0, 0)
    scale = float(target_size) / max(w, h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    img_resized = img.resize((nw, nh), Image.BILINEAR)
    canvas = Image.new(img.mode, (target_size, target_size))
    left = (target_size - nw) // 2
    top  = (target_size - nh) // 2
    canvas.paste(img_resized, (left, top))
    return canvas, scale, (left, top)

def polygons_to_mask(polygons: List[List[float]], mask_size: Tuple[int,int]) -> np.ndarray:
    """
    polygons: list of flat [x1,y1,x2,y2,...]
    Returns binary mask (H,W). Uses PIL to rasterize polygons robustly.
    """
    w, h = mask_size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        if len(poly) >= 6:
            pts = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            draw.polygon(pts, outline=1, fill=1)
    return np.array(mask, dtype=np.uint8)

def transform_points(points: List[float], scale: float, pad: Tuple[int,int]) -> List[float]:
    """Apply scale + letterbox pad to polygon points."""
    left, top = pad
    out = []
    for i, v in enumerate(points):
        if i % 2 == 0:
            # x
            out.append(v * scale + left)
        else:
            # y
            out.append(v * scale + top)
    return out

def transform_bbox(bbox: List[float], scale: float, pad: Tuple[int,int]) -> List[float]:
    """
    bbox = [x, y, w, h] in original coords.
    Returns [x, y, w, h] in resized+letterboxed coords.
    """
    x, y, w, h = bbox
    left, top = pad
    x2 = x + w
    y2 = y + h
    x = x * scale + left
    y = y * scale + top
    x2 = x2 * scale + left
    y2 = y2 * scale + top
    return [x, y, max(0.0, x2 - x), max(0.0, y2 - y)]

def to_torch_image(img: Image.Image, normalize: bool = True) -> "torch.Tensor":
    """Convert PIL -> FloatTensor (C,H,W), [0,1], optional mean/std normalize."""
    arr = np.array(img).astype(np.float32)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    arr = arr / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        tensor = (tensor - mean) / std
    return tensor


#Core preprocessing
def process_image_record(
    img_rec: Dict[str, Any],
    anns: List[Dict[str, Any]],
    images_dir: str,
    target_size: Optional[int] = None,
    build_semantic: bool = False,
    normalize: bool = True,
    return_flatten: bool = False
) -> Dict[str, Any]:
    """
    Returns a dict with:
    - image_tensor (torch.Tensor) or image_array (np.ndarray) if torch unavailable
    - flat_vector (optional): flattened pixel vector
    - instance_masks: np.ndarray [N,H,W] (uint8)
    - semantic_mask (optional): np.ndarray [H,W] (int32 of category_id)
    - bboxes: list[[x,y,w,h]]
    - categories: list[int]
    - areas: list[float]
    - iscrowd: list[int]
    - attributes: list[dict] (if present)
    - meta: {file_name, orig_size, size}
    """
    file_name = img_rec["file_name"]
    w0, h0 = img_rec["width"], img_rec["height"]
    path = os.path.join(images_dir, file_name)
    img = Image.open(path).convert("RGB")

    #Resize + letterbox
    img_lb, scale, pad = resize_and_letterbox(img, target_size)
    W, H = img_lb.size

    #Build instance masks and targets
    instance_masks = []
    bboxes = []
    categories = []
    areas = []
    iscrowd = []
    attributes = []

    for ann in anns:
        #Segmentation: list of polygons (each polygon is a flat list)
        seg = ann.get("segmentation", [])
        polys_resized = [transform_points(poly, scale, pad) for poly in seg]
        mask = polygons_to_mask(polys_resized, (W, H))
        if mask.sum() == 0:
            #Skip empty masks after resize (rare but possible)
            continue
        instance_masks.append(mask)
        #Bbox
        bbox = transform_bbox(ann.get("bbox", [0,0,0,0]), scale, pad)
        bboxes.append(bbox)
        categories.append(int(ann.get("category_id", -1)))
        areas.append(float(ann.get("area", float(mask.sum()))))
        iscrowd.append(int(ann.get("iscrowd", 0)))
        attributes.append(ann.get("attributes", {}))

    if len(instance_masks) > 0:
        instance_masks = np.stack(instance_masks, axis=0).astype(np.uint8)
    else:
        instance_masks = np.zeros((0, H, W), dtype=np.uint8)

    #semantic mask (category_id per pixel; 0 = background)
    semantic_mask = None
    if build_semantic:
        semantic_mask = np.zeros((H, W), dtype=np.int32)
        for m, c in zip(instance_masks, categories):
            semantic_mask[m.astype(bool)] = c

    #Convert image
    if torch is not None:
        image_tensor = to_torch_image(img_lb, normalize=normalize)
        image_array = None
    else:
        arr = np.array(img_lb).astype(np.float32) / 255.0
        image_tensor = None
        image_array = arr

    #flatten
    flat_vector = None
    if return_flatten:
        if torch is not None:
            flat_vector = image_tensor.flatten().detach().cpu().numpy()
        else:
            flat_vector = image_array.reshape(-1)

    return {
        "image_tensor": image_tensor,
        "image_array": image_array,
        "flat_vector": flat_vector,
        "instance_masks": instance_masks,
        "semantic_mask": semantic_mask,
        "bboxes": bboxes,
        "categories": categories,
        "areas": areas,
        "iscrowd": iscrowd,
        "attributes": attributes,
        "meta": {
            "file_name": file_name,
            "orig_size": (w0, h0),
            "size": (W, H),
            "scale": scale,
            "pad": pad
        }
    }


# PyTorch Dataset 
class CarDDDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        annotations_path: str,
        target_size: Optional[int] = 512,
        build_semantic: bool = False,
        normalize: bool = True,
        return_flatten: bool = False,
        keep_ids: Optional[set] = None
    ):
        if torch is None:
            raise RuntimeError("Torch is required for the Dataset. Install torch or use --save_numpy mode.")
        self.images_dir = images_dir
        self.coco = load_coco(annotations_path)
        self.images_by_id, self.anns_by_img, self.cat_name = build_indices(self.coco)
        #Filter image ids if a subset is requested
        self.image_ids = [img_id for img_id in self.images_by_id.keys()
                          if (keep_ids is None or img_id in keep_ids)]
        self.target_size = target_size
        self.build_semantic = build_semantic
        self.normalize = normalize
        self.return_flatten = return_flatten

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx: int):
        img_id = self.image_ids[idx]
        img_rec = self.images_by_id[img_id]
        anns = self.anns_by_img.get(img_id, [])
        rec = process_image_record(
            img_rec, anns, self.images_dir,
            target_size=self.target_size,
            build_semantic=self.build_semantic,
            normalize=self.normalize,
            return_flatten=self.return_flatten
        )
        #Package targets similar to torchvision detection
        target = {
            "boxes": torch.tensor(rec["bboxes"], dtype=torch.float32) if rec["bboxes"] else torch.zeros((0,4), dtype=torch.float32),
            "labels": torch.tensor(rec["categories"], dtype=torch.int64) if rec["categories"] else torch.zeros((0,), dtype=torch.int64),
            "masks": torch.from_numpy(rec["instance_masks"]).to(torch.uint8) if rec["instance_masks"].size else torch.zeros((0, *rec["instance_masks"].shape[-2:]), dtype=torch.uint8),
            "iscrowd": torch.tensor(rec["iscrowd"], dtype=torch.int64) if rec["iscrowd"] else torch.zeros((0,), dtype=torch.int64),
            "areas": torch.tensor(rec["areas"], dtype=torch.float32) if rec["areas"] else torch.zeros((0,), dtype=torch.float32),
            "image_id": torch.tensor([img_id], dtype=torch.int64),
            "meta": rec["meta"]
        }
        if rec["semantic_mask"] is not None:
            target["semantic_mask"] = torch.from_numpy(rec["semantic_mask"]).to(torch.int64)
        if rec["attributes"]:
            target["attributes"] = rec["attributes"]
        out = {
            "image": rec["image_tensor"],
            "flat_vector": torch.from_numpy(rec["flat_vector"]).float() if rec["flat_vector"] is not None else None,
            "target": target
        }
        return out

def collate_fn(batch):
    #For detection tasks with variable sizes
    images = [b["image"] for b in batch]
    targets = [b["target"] for b in batch]
    flat = [b["flat_vector"] for b in batch if b["flat_vector"] is not None]
    return images, targets, flat if flat else None

#Numpy dump 
def dump_numpy_bundle(
    images_dir: str,
    annotations_path: str,
    out_dir: str,
    img_size: Optional[int],
    build_semantic: bool,
    save_flat_vectors: bool
):
    coco = load_coco(annotations_path)
    images_by_id, anns_by_img, cat_name = build_indices(coco)
    image_ids = list(images_by_id.keys())

    X = []
    X_flat = []
    all_masks = []
    sem_masks = []
    targets = {}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_id in image_ids:
        rec = images_by_id[img_id]
        anns = anns_by_img.get(img_id, [])
        processed = process_image_record(
            rec, anns, images_dir,
            target_size=img_size,
            build_semantic=build_semantic,
            normalize=False,            # keep raw 0..1 for Numpy bundle
            return_flatten=save_flat_vectors
        )
        # Save image array as HWC float32
        if processed["image_array"] is None:
            # torch-only path; convert back to HWC for numpy bundle
            arr = processed["image_tensor"].permute(1,2,0).cpu().numpy()
        else:
            arr = processed["image_array"]
        X.append(arr.astype(np.float32))

        if save_flat_vectors:
            X_flat.append(processed["flat_vector"].astype(np.float32))

        # Save instance masks
        all_masks.append(processed["instance_masks"])  # shape (N,H,W)

        if build_semantic:
            sem_masks.append(processed["semantic_mask"])

        targets[img_id] = {
            "file_name": processed["meta"]["file_name"],
            "size": processed["meta"]["size"],
            "orig_size": processed["meta"]["orig_size"],
            "boxes": processed["bboxes"],
            "labels": processed["categories"],
            "areas": processed["areas"],
            "iscrowd": processed["iscrowd"],
            "attributes": processed["attributes"]
        }

    # Stack/sync variable-length masks safely into npz
    X = np.stack(X, axis=0)  # (B,H,W,3)
    np.save(out_dir / "images.npy", X)
    if save_flat_vectors:
        X_flat = np.stack(X_flat, axis=0)
        np.save(out_dir / "flat_vectors.npy", X_flat)

    # Masks: different #instances per image -> save as list in npz
    np.savez_compressed(
        out_dir / "instance_masks.npz",
        masks=np.array(all_masks, dtype=object)
    )
    if build_semantic:
        sem_masks = np.stack(sem_masks, axis=0)
        np.save(out_dir / "semantic_masks.npy", sem_masks)

    with open(out_dir / "targets.json", "w") as f:
        json.dump({"targets": targets, "categories": cat_name}, f, indent=2)


# CLI
def parse_args():
    ap = argparse.ArgumentParser("CAR-DD Preprocessor")
    ap.add_argument("--images_dir", required=True, help="Path to images root")
    ap.add_argument("--annotations", required=True, help="Path to COCO-style JSON")
    ap.add_argument("--out_dir", required=True, help="Where to write outputs")
    ap.add_argument("--img_size", type=int, default=512, help="Square letterbox size; set to 0 to keep original")
    ap.add_argument("--no_letterbox", action="store_true", help="Use original size (no resize/letterbox)")
    ap.add_argument("--build_semantic", action="store_true", help="Also build per-pixel category map")
    ap.add_argument("--save_numpy", action="store_true", help="Export Numpy bundles")
    ap.add_argument("--save_flat_vectors", action="store_true", help="Also save flattened pixel vectors")
    return ap.parse_args()

def main():
    args = parse_args()
    img_size = None if args.no_letterbox or args.img_size == 0 else args.img_size
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # dump numpy bundles for classical ML pipelines
    if args.save_numpy:
        dump_numpy_bundle(
            images_dir=args.images_dir,
            annotations_path=args.annotations,
            out_dir=args.out_dir,
            img_size=img_size,
            build_semantic=args.build_semantic,
            save_flat_vectors=args.save_flat_vectors
        )

    print("Done.")
    print(f"- Outputs in: {args.out_dir}")
    if args.save_numpy:
        print("  Saved: images.npy, instance_masks.npz, targets.json", end="")
        if args.build_semantic:
            print(", semantic_masks.npy", end="")
        if args.save_flat_vectors:
            print(", flat_vectors.npy", end="")
        print("")

if __name__ == "__main__":
    main()
