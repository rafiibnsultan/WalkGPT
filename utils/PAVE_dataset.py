# === NEW: text-only dataset built to match your collate_fn ===
import json, re, os
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from model.llava_walkgpt import conversation as conversation_lib
from transformers import CLIPImageProcessor
from model.segment_anything.utils.transforms import ResizeLongestSide

from .utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_TOKEN)

def _strip_assessment_tags(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r"^\s*<assessment>\s*", "", s.strip(), flags=re.IGNORECASE)
    s = re.sub(r"\s*</assessment>\s*$", "", s.strip(), flags=re.IGNORECASE)
    return s.strip()

def _depth_to_string(depth_dict: dict, decimals: int = 2) -> str:
    # keys may be strings; sort numerically
    items = sorted(depth_dict.items(), key=lambda kv: int(kv[0]))
    parts = [f"id {int(k)}: {float(v):.{decimals}f} m" for k, v in items]
    return ", ".join(parts)

def _maybe_read(path_no_ext: str, exts=(".jpg", ".png", ".jpeg")):
    """Return first existing full path for any of the given extensions, else None."""
    for e in exts:
        p = f"{path_no_ext}{e}"
        if os.path.isfile(p):
            return p
    return None


# --- EDITED DATASET (returns 10 items like SemSegDataset) ---
class PAVEDataset(torch.utils.data.Dataset):
    """
    Emits the SAME 10-tuple as SemSegDataset.__getitem__:
      (image_path, image, image_clip, conversations, masks, label, resize, clip_resize, questions, sampled_classes)

    It reads:
      - image  from: {session}/video_frames/{index}.{jpg|png}
      - mask   from: {session}/masks/{index}.pt   (if present; treated as semantic labels)
      - text   from: assessment + depth dict
    """
    pixel_mean = torch.Tensor([97.17, 105.73, 108.16]).view(-1, 1, 1)
    pixel_std  = torch.Tensor([53.05, 56.40, 61.93]).view(-1, 1, 1)
    img_size   = 1024  # for the final padded square like SemSegDataset
    ignore_label = 255

    def __init__(
        self,
        jsonl_path: str = "./datasets/train_85.jsonl",  # Minimal annotations: the answer field only contains the textual assessment. During preprocessing, the pipeline augments each sample by attaching depth, segmentation masks, and other spatial metadata from the dataset annotations.
        labelmap_path: str = "./datasets/labelmap.json", # full dataset link: https://huggingface.co/datasets/rafiibnsultan1/PAVE
        accessible_threshold: float = 0.5,
        tokenizer=None,                     # kept for parity (unused here)
        vision_tower: str = "openai/clip-vit-large-patch14",
        samples_per_epoch: int | None = None,  # if None, use actual len
        precision: str = "fp32",
        image_size: int = 224,              # shortest-edge for SAM resize
        seg_token_num: int = 1,             # parity only
        pad_train_clip_images: bool = False,
        masks_process_with_clip: bool = False,
        preprocessor_config: str = "",      # custom CLIPImageProcessor checkpoint
    ):
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.precision = precision
        self.seg_token_num = max(1, int(seg_token_num))
        if self.seg_token_num == 1:
            self._seg_token_list = ["[SEG]"]
        else:
            self._seg_token_list = [f"[SEG{i}]" for i in range(self.seg_token_num)]
        self._seg_token_marker = " ".join(self._seg_token_list)
        self.image_size = image_size
        self.pad_train_clip_images = pad_train_clip_images
        self.masks_process_with_clip = masks_process_with_clip
        self.samples_per_epoch = samples_per_epoch  # may be None

        # image preprocessors (match SemSegDataset)
        self.transform      = ResizeLongestSide(image_size)  # for SAM path
        self.clip_processor = (CLIPImageProcessor.from_pretrained(vision_tower)
                               if preprocessor_config == ''
                               else CLIPImageProcessor.from_pretrained(preprocessor_config))
        self.transform_clip = ResizeLongestSide(self.clip_processor.size['shortest_edge'])

        # load jsonl
        self.samples = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    self.samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                
        self.accessible_threshold = accessible_threshold
        # Load label map once
        with open(labelmap_path, "r") as f:
            lm = json.load(f)
        # Expect: {"id_to_name": {...}, "accessibility_scores": {...}}
        self.id_to_name = lm.get("id_to_name", {})           # keys as strings
        self.name_to_score = lm.get("accessibility_scores", {})

    def __len__(self):
        # return len(self.samples)
        return self.samples_per_epoch if self.samples_per_epoch is not None else len(self.samples)

    # same normalization/pad as SemSegDataset.preprocess
    def preprocess(self, x: torch.Tensor, decoder_image_size: int) -> torch.Tensor:
        x = (x - self.pixel_mean) / self.pixel_std
        h, w = x.shape[-2:]
        padh = decoder_image_size - h
        padw = decoder_image_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x

    def _compose_text(self, ex: dict) -> str:
        """
        Build the final assistant text with:
        - [assessment]...[/assessment]
        - Accessible / Non-accessible feature lists as [p]{name}[/p] separated by [SEG]
        - [distance] ... [/distance] listing id_to_name and distances (m), sorted by distance asc
        Rules:
        - Only include classes present in ex['depth']
        - Map ids -> names via self.id_to_name; skip ids we can't map
        - Accessibility via self.name_to_score[name] >= self.accessible_threshold
        - Distances formatted with 1 decimal place
        """
        # 1) Assessment (keeps original tags, or re-wraps clean text)
        raw_assess = ex.get("assessment", "")
        assess_text = _strip_assessment_tags(raw_assess)
        assess_block = f"[assessment] {assess_text} [/assessment]"

        # 2) Collect (id, name, dist) triples from depth dict
        depth = ex.get("depth", {}) or {}
        triples = []
        for k, v in depth.items():
            sid = str(k)
            name = self.id_to_name.get(sid, None)
            if name is None:
                continue
            try:
                dist = float(v)
            except Exception:
                continue
            triples.append((int(sid), name, dist))

        if not triples:
            # If nothing to map, just return the assessment block
            return assess_block

        # 3) Split into accessible vs non-accessible by score threshold
        acc_ids, nonacc_ids = [], []
        for _, name, _ in triples:
            s = self.name_to_score.get(name, None)
            if s is None:
                # Skip items missing a score (per your earlier rule)
                continue
            (acc_ids if s >= self.accessible_threshold else nonacc_ids).append(name)

        # Keep deterministic ordering (alphabetical)
        acc_ids = sorted(set(acc_ids))
        nonacc_ids = sorted(set(nonacc_ids))

        seg_marker = self._seg_token_marker

        def _pack_features(names: list[str]) -> str:
            if not names:
                return ""
            return "".join([f"[p] {n} [/p]{seg_marker}" for n in names])

        acc_block = f" Accessible features are here: {_pack_features(acc_ids)}" if acc_ids else ""
        nonacc_block = f" Non-accessible features are here: {_pack_features(nonacc_ids)}" if nonacc_ids else ""

        # 4) Distance block: sort by ascending distance and render
        # Only list distances for ids we can map to names (already filtered in triples)
        triples_sorted = sorted(triples, key=lambda x: x[2])
        parts = []
        for _, name, d in triples_sorted:
            parts.append(f"to the {name}: {d:.1f} m")
        dist_body = "; ".join(parts)
        dist_block = f" [distance] Distance from the user to the {dist_body}. [/distance]" if parts else ""

        # 5) Final string
        return f"{assess_block}{acc_block}{nonacc_block}{dist_block}"


    def __getitem__(self, idx):
        # allow epoch sampling greater than true len (like SemSegDataset’s random access)
        if idx >= len(self.samples):
            idx = np.random.randint(0, len(self.samples))
        ex = self.samples[idx]

        # ---------- find image path ----------
        session = ex.get("session", "").rstrip("/")
        index   = ex.get("index", "")
        vf_root = os.path.join(session, "video_frames", index)
        img_path = _maybe_read(vf_root)  # try .jpg/.png/.jpeg
        if img_path is None:
            # keep a plausible path string for traceability
            img_path = f"{vf_root}.png"

        # ---------- load image (or dummy) ----------
        if os.path.isfile(img_path):
            img_bgr = cv2.imread(img_path)
            image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # dummy
            image_rgb = np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)

        # ---------- CLIP image ----------
        if self.pad_train_clip_images:
            image_clip_np = self.transform_clip.apply_image(image_rgb)
            clip_resize = image_clip_np.shape[:2]
            image_clip = self.preprocess(
                torch.from_numpy(image_clip_np).permute(2,0,1).contiguous().float(),
                self.clip_processor.size['shortest_edge']
            )
        else:
            # standard CLIPProcessor path (already normalized to CLIP, not our mean/std)
            image_clip = self.clip_processor.preprocess(image_rgb, return_tensors="pt")["pixel_values"][0]
            clip_resize = image_clip.shape[-2:]

        # ---------- SAM-side resize & our normalize/pad ----------
        image_np = self.transform.apply_image(image_rgb)  # shortest edge = image_size
        resize = image_np.shape[:2]
        image = self.preprocess(
            torch.from_numpy(image_np).permute(2,0,1).contiguous().float(),
            self.img_size
        )

        # ---------- load semantic label mask if present ----------
        # Expecting a per-pixel semantic PNG under segmentation_masks/{index}.png|jpg
        # ---------- semantic label from .pt ----------
        # expected file: {session}/masks/{index}.pt  (or .pth fallback)
        msk_root = os.path.join(session, "masks", index)
        pt_candidates = [f"{msk_root}.pt", f"{msk_root}.pth"]

        mask_path = next((p for p in pt_candidates if os.path.isfile(p)), None)

        if mask_path:
            try:
                loaded = torch.load(mask_path, map_location="cpu", weights_only="True")

                # Accept common variants: tensor, dict with known keys, numpy
                if isinstance(loaded, dict):
                    # try a few common keys
                    for k in ("mask", "semantic_mask", "label", "labels"):
                        if k in loaded:
                            loaded = loaded[k]
                            break

                if isinstance(loaded, np.ndarray):
                    label_t = torch.from_numpy(loaded)
                else:
                    label_t = loaded if isinstance(loaded, torch.Tensor) else None

                if label_t is None:
                    raise ValueError(f"Unrecognized mask payload type: {type(loaded)}")

                # ensure shape is (H, W)
                if label_t.ndim == 3 and label_t.shape[0] == 1:
                    label_t = label_t.squeeze(0)
                if label_t.ndim == 3 and label_t.shape[-1] == 1:
                    label_t = label_t.squeeze(-1)
                if label_t.ndim != 2:
                    raise ValueError(f"Mask tensor must be (H,W); got {tuple(label_t.shape)}")

                # resize to match SAM-resized image 'resize=(H,W)' using nearest
                # label_t: (H0, W0) -> (H, W)
                label_t = label_t.to(torch.long)
                label_t = F.interpolate(
                    label_t.unsqueeze(0).unsqueeze(0).float(),  # (1,1,H0,W0)
                    size=(resize[0], resize[1]),
                    mode="nearest"
                ).squeeze(0).squeeze(0).to(torch.long)

                label = label_t  # final semantic label (H, W) long

            except Exception as e:
                # fallback to ignore plane on any read/format error
                # (H, W) from 'resize'
                label = torch.full((resize[0], resize[1]), self.ignore_label, dtype=torch.long)
        else:
            # no label available
            label = torch.full((resize[0], resize[1]), self.ignore_label, dtype=torch.long)

        # ---------- masks tensor ----------
        # Provide a single-plane validity mask if label exists (like a generic foreground).
        # If you later want per-class planes, we can expand by unique IDs ≤ 30.
        # ex["depth"] has string keys like {"16": 29.48, "29": 1.98, "30": 2.91}
        depth_dict = ex.get("depth", {}) or {}
        sampled_ids = sorted(int(k) for k in depth_dict.keys())
        sampled_classes = [[str(cid)] for cid in sampled_ids]  # [["16"], ["29"], ["30"], ...]

        # Build one (H, W) binary mask per sampled class id from the semantic label.
        if (label != self.ignore_label).any() and sampled_ids:
            planes = [(label == cid).to(torch.float32) for cid in sampled_ids]
            masks = torch.stack(planes, dim=0)   # (N_classes, H, W)
        else:
            masks = torch.zeros(0, resize[0], resize[1], dtype=torch.float32)

        # Optionally align masks to CLIP square (like SemSegDataset does when masks_process_with_clip)
        if self.masks_process_with_clip:
            mask_shape = image_clip.shape[-1]
            if masks.numel() == 0:
                masks = torch.zeros(0, mask_shape, mask_shape)
            else:
                # nearest resize then center-crop to (mask_shape, mask_shape)
                # First, bring masks to current (H,W) explicitly (no-op if already)
                masks = F.interpolate(masks.unsqueeze(0), size=(resize[0], resize[1]), mode="nearest").squeeze(0)

                h, w = masks.shape[-2:]
                short, long = (w, h) if w <= h else (h, w)
                new_short = mask_shape
                new_long  = int(new_short * long / short)
                new_shape = (new_long, new_short) if w <= h else (new_short, new_long)

                masks = F.interpolate(masks.unsqueeze(0).float(), size=new_shape, mode="nearest").squeeze(0).bool()
                oh, ow = new_shape
                top  = (oh - mask_shape) // 2
                left = (ow - mask_shape) // 2
                masks = masks[:, top:top + mask_shape, left:left + mask_shape].to(torch.float32)

        # ex["depth"] has string keys like {"16": 29.48, "29": 1.98, "30": 2.91}
        depth_dict = ex.get("depth", {}) or {}
        # sort by numeric class id for determinism
        sampled_ids = sorted([int(k) for k in depth_dict.keys()])

        # Represent sampled_classes the same shape SemSegDataset downstream expects:
        # a list of per-question class lists. Since we have no grouping logic here,
        # we return singletons: [["16"], ["29"], ["30"], ...] as strings.
        sampled_classes = [[str(cid)] for cid in sampled_ids]

        # ---------- compose assistant text once ----------
        assistant_resp = self._compose_text(ex)

        # ---------- normalize questions from JSON ----------
        q_raw = ex.get("question", None)
        if isinstance(q_raw, list):
            questions_clean = [str(q).strip() for q in q_raw if str(q).strip()]
        elif isinstance(q_raw, str) and q_raw.strip():
            questions_clean = [q_raw.strip()]
        else:
            questions_clean = [
                "Which nearby features seem pedestrian-friendly, and which could make movement unsafe?"
            ]

        questions_prefixed = [f"{DEFAULT_IMAGE_TOKEN}\n{q}" for q in questions_clean]

        # ---------- build conversations (one per question) ----------
        conversations = []
        for q in questions_prefixed:
            conv = conversation_lib.default_conversation.copy()
            conv.messages = []
            conv.append_message(conv.roles[0], q)              # Human
            conv.append_message(conv.roles[1], assistant_resp) # Assistant
            conversations.append(conv.get_prompt())

        # ---------- package question metadata ----------
        target_counts = [len(cls_ids) for cls_ids in sampled_classes] if sampled_classes else []
        category_names = []
        for cls_ids in sampled_classes:
            names = []
            for cls_id in cls_ids:
                names.append(self.id_to_name.get(str(cls_id), str(cls_id)))
            category_names.append(names)

        if depth_dict:
            dist_parts = []
            for cid in sampled_ids:
                name = self.id_to_name.get(str(cid), str(cid))
                dist_val = depth_dict.get(str(cid), None)
                if dist_val is None:
                    dist_parts.append(name)
                else:
                    dist_parts.append(f"{name} at {float(dist_val):.1f} m")
            prompt_ins = "PAVE depth cues: " + ", ".join(dist_parts)
        else:
            prompt_ins = "PAVE depth cues: none available."

        questions_payload = (questions_prefixed, target_counts, category_names, prompt_ins)


        # Return EXACT 10-tuple like SemSegDataset
        return (
            img_path,       # image_path (string)
            image,          # (3, 1024, 1024) float32 after normalize+pad
            image_clip,     # (3, Hc, Wc) CLIP pixel_values
            conversations,  # list[str], length == len(questions)
            masks,          # (N_classes, H, W) float32
            label,          # (H, W) long semantic ids (or ignore plane)
            resize,         # (H, W) after SAM resize
            clip_resize,    # (Hc, Wc)
            questions_payload,  # tuple: (questions, target_counts, category_names, prompt_ins)
            sampled_classes # list[list[str]], one entry per class plane/order
        )
        
        
# --- Validation dataset that mirrors PAVEDataset outputs (10-tuple) ---
class PAVEValDataset(PAVEDataset):
    """
    Same output arity and fields as PAVEDataset:
      (image_path, image, image_clip, conversations, masks, label, resize, clip_resize, questions, sampled_classes)

    Differences from train:
      - No samples_per_epoch oversampling (uses true length).
      - Optional knobs for clip padding / mask alignment, but defaults match your train dataset’s behavior.
      - Inherits _compose_text and all preprocessing paths from PAVEDataset.
    """
    def __init__(
        self,
        jsonl_path: str = "./datasets/val_85.jsonl",
        labelmap_path: str = "./datasets/labelmap.json", 
        accessible_threshold: float = 0.5,
        tokenizer=None,                     # kept for parity (unused here)
        vision_tower: str = "openai/clip-vit-large-patch14",
        samples_per_epoch: int | None = None,  # if None, use actual len
        precision: str = "fp32",
        image_size: int = 224,              # shortest-edge for SAM resize
        seg_token_num: int = 1,             # parity only
        pad_val_clip_images: bool = False,
        masks_process_with_clip: bool = False,
        preprocessor_config: str = "",      # custom CLIPImageProcessor checkpoint
    ):
        super().__init__(
            jsonl_path=jsonl_path,
            labelmap_path=labelmap_path,
            accessible_threshold=accessible_threshold,
            tokenizer=tokenizer,
            vision_tower=vision_tower,
            samples_per_epoch=None,            # <- val uses actual len
            precision=precision,
            image_size=image_size,
            seg_token_num=seg_token_num,
            pad_train_clip_images=pad_val_clip_images,
            masks_process_with_clip=masks_process_with_clip,
            preprocessor_config=preprocessor_config,
        )
        # (Optional) any val-only switches go here.
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # Get the original 10-tuple from PAVEDataset
        (
            image_path,       # str
            image,            # (3, 1024, 1024) float32
            image_clip,       # (3, Hc, Wc)
            conversations,    # list[str]
            masks,            # (N, H, W) float32
            label,            # (H, W) long
            resize,           # (H, W)
            clip_resize,      # (Hc, Wc)
            questions,        # list[str]
            sampled_classes,  # list[list[str]]
        ) = super().__getitem__(idx)

        # Val-style fields
        labels = label                         # keep semantic ids / ignore plane
        inference = True                       # val/inference mode
        # Two placeholders + a boolean as required by your ValDataset signature:
        return (
            image_path,
            image,
            image_clip,
            conversations,
            masks,
            labels,
            resize,
            clip_resize,
            questions,
            sampled_classes,
            False,
            inference,
        )
