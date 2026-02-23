from enum import Enum

import numpy as np
import torch
import torch.distributed as dist

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

SHORT_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you segment the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please segment the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "What is {class_name} in this image? Please output segmentation mask.",
]

LONG_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please respond with segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please output segmentation mask.",
]

EXPLANATORY_QUESTION_LIST = [
    "Please output segmentation mask and explain why.",
    "Please output segmentation mask and explain the reason.",
    "Please output segmentation mask and give some explanation.",
]

ANSWER_LIST = [
    "It is [SEG].",
    "Sure, [SEG].",
    "Sure, it is [SEG].",
    "Sure, the segmentation result is [SEG].",
    "[SEG].",
]

SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
    "The segmentation result of {class_name} is [SEG].",
    "[SEG]."
]

MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
    "Sure, they are {seg}.",
    "They are {seg}.",
    "{seg}."
]

MR_SINGLE_ANSWER_LIST = [
    "{class_name} is [SEG].",
]

MR_MULTI_ANSWER_LIST = [
    "{class_name} are {seg}, separately.",
    "{class_name} are {seg}.",
    "Sure, {class_name} are {seg}, separately.",
    "Sure, {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}.",
    "the segmentation result of {class_name} are {seg}, separately.",
    "Sure, the segmentation result of {class_name} are {seg}.",
    "Sure, the segmentation result of {class_name} are {seg}, separately.",
]


EXPAND_LONG_QUESTION_LIST = [

    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Provide the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Output the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please show the segmentation mask.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} I'd appreciate segmentation masks.",
    DEFAULT_IMAGE_TOKEN + "\n" + "{sent} Please highlight the segmentation mask.",

]

EXPAND_QUESTION_LIST = [
    DEFAULT_IMAGE_TOKEN + "\n" + "Could you identify the {class_name} in this picture?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Are you able to delineate the {class_name} in the image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you pinpoint the {class_name} in this photo?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Is it possible for you to highlight the {class_name} in this image?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Can you discern the {class_name} in the given picture?",

    DEFAULT_IMAGE_TOKEN + "\n" + "Can you provide me with asegment of the {class_name}?",
    DEFAULT_IMAGE_TOKEN + "\n" + "Please perform image segmentation to isolate the {class_name} in this image.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Help me segment the {class_name}.",
    DEFAULT_IMAGE_TOKEN + "\n" + "Would you be willing to segment the {class_name}?",
    

    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Can you identify {class_name} in this picture? Please provide a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Could you point out {class_name} in this image and show it with a segmentation mask?",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In this image, where is {class_name}? I'd appreciate a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "Please highlight {class_name} in this image using a segmentation mask.",
    DEFAULT_IMAGE_TOKEN
    + "\n"
    + "In the picture provided, can you show where {class_name} is with a segmentation mask?",
    
]

class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if isinstance(self.sum, np.ndarray):
            total = torch.tensor(
                self.sum.tolist()
                + [
                    self.count,
                ],
                dtype=torch.float32,
                device=device,
            )
        else:
            total = torch.tensor(
                [self.sum, self.count], dtype=torch.float32, device=device
            )

        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        if total.shape[0] > 2:
            self.sum, self.count = total[:-1].cpu().numpy(), total[-1].cpu().item()
        else:
            self.sum, self.count = total.tolist()
        self.avg = self.sum / (self.count + 1e-5)

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.dim() in [1, 2, 3]
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K - 1)
    area_output = torch.histc(output, bins=K, min=0, max=K - 1)
    area_target = torch.histc(target, bins=K, min=0, max=K - 1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def dict_to_cuda(input_dict):
    for k, v in input_dict.items():
        if isinstance(input_dict[k], torch.Tensor):
            input_dict[k] = v.cuda(non_blocking=True)
        elif (
            isinstance(input_dict[k], list)
            and len(input_dict[k]) > 0
            and isinstance(input_dict[k][0], torch.Tensor)
        ):
            input_dict[k] = [ele.cuda(non_blocking=True) for ele in v]
    return input_dict



# Canonical class names from your dataset
CANONICAL = {
    "unlabeled", "road", "curb", "sidewalk", "guard rail/road barrier",
    "crosswalk", "paved trail", "building", "wall/fence", "hand rail",
    "opening-door", "opening-gate", "pedestrian", "rider", "animal",
    "stairs", "water body", "other walkable surface", "inaccessible surface",
    "railway track", "obstacle", "vehicle", "traffic sign", "traffic light",
    "pole", "bus stop", "bike rack", "sky", "tree", "vegetation", "terrain"
}

# Targeted aliases -> canonical
ALIASES = {
    # exact canonical kept for completeness
    "unlabeled": "unlabeled",
    "road": "road",
    "curb": "curb",
    "sidewalk": "sidewalk",
    "guard rail/road barrier": "guard rail/road barrier",
    "crosswalk": "crosswalk",
    "paved trail": "paved trail",
    "building": "building",
    "wall/fence": "wall/fence",
    "hand rail": "hand rail",
    "opening-door": "opening-door",
    "opening-gate": "opening-gate",
    "pedestrian": "pedestrian",
    "rider": "rider",
    "animal": "animal",
    "stairs": "stairs",
    "water body": "water body",
    "other walkable surface": "other walkable surface",
    "inaccessible surface": "inaccessible surface",
    "railway track": "railway track",
    "obstacle": "obstacle",
    "vehicle": "vehicle",
    "traffic sign": "traffic sign",
    "traffic light": "traffic light",
    "pole": "pole",
    "bus stop": "bus stop",
    "bike rack": "bike rack",
    "sky": "sky",
    "tree": "tree",
    "vegetation": "vegetation",
    "terrain": "terrain",

    # common variants / spacing / punctuation
    "side walk": "sidewalk",
    "side-walk": "sidewalk",

    "guard rail": "guard rail/road barrier",
    "guardrail": "guard rail/road barrier",
    "guard-rail": "guard rail/road barrier",
    "road barrier": "guard rail/road barrier",
    "road-barrier": "guard rail/road barrier",

    "pavement": "paved trail",            # if you prefer keep as sidewalk, change here
    "paved-trail": "paved trail",

    "wall": "wall/fence",
    "fence": "wall/fence",
    "wall / fence": "wall/fence",
    "wall- fence": "wall/fence",
    "fence/wall": "wall/fence",

    "handrail": "hand rail",
    "hand-rail": "hand rail",

    "opening door": "opening-door",
    "open door": "opening-door",
    "door opening": "opening-door",

    "opening gate": "opening-gate",
    "open gate": "opening-gate",
    "gate opening": "opening-gate",

    "pedestrians": "pedestrian",

    "riders": "rider",

    "animals": "animal",

    "stair": "stairs",
    "staircase": "stairs",
    "staircases": "stairs",
    "stairs case": "stairs",

    "waterbody": "water body",
    "water-body": "water body",
    "water": "water body",

    "other walkable": "other walkable surface",
    "walkable surface": "other walkable surface",

    "inaccessible": "inaccessible surface",
    "non-accessible surface": "inaccessible surface",

    "railway": "railway track",
    "rail track": "railway track",
    "railroad track": "railway track",
    "train track": "railway track",

    "obstacles": "obstacle",

    "vehicles": "vehicle",
    "car": "vehicle",
    "cars": "vehicle",
    "truck": "vehicle",
    "trucks": "vehicle",
    "bus": "vehicle",          # note: “bus stop” handled separately below
    "bicycle": "vehicle",      # if you’d rather map to bike rack only when ‘rack’ present, keep as is
    "bike": "vehicle",

    "traffic signs": "traffic sign",
    "sign": "traffic sign",        # cautious: remove if too broad
    "signs": "traffic sign",

    "traffic lights": "traffic light",
    "signal": "traffic light",     # cautious
    "signals": "traffic light",

    "poles": "pole",

    "bus-stop": "bus stop",
    "bus station": "bus stop",     # cautious

    "bike-rack": "bike rack",
    "bicycle rack": "bike rack",

    "trees": "tree",
    "bush": "vegetation",
    "bushes": "vegetation",
    "plants": "vegetation",
    "shrub": "vegetation",
    "shrubs": "vegetation",

    "ground": "terrain",
    "dirt": "terrain",
    "grass": "terrain",            # if you treat grass as vegetation, change to "vegetation"
}

import json
import re
import math
import numpy as np
from collections import Counter

def canonicalize_obj(name: str) -> str:
    """
    Normalize raw object labels to your dataset’s canonical set.
    Conservative: only map when confident; otherwise return a cleaned string.
    """
    if not name:
        return ""

    s = name.strip().lower()
    # squeeze spaces
    s = re.sub(r"\s+", " ", s)
    # normalize slashes and hyphens spacing
    s = s.replace(" / ", "/").replace(" /", "/").replace("/ ", "/")
    s = s.replace(" - ", "-").strip()

    # quick wins
    if s in ALIASES:
        return ALIASES[s]

    # remove trailing plural 's' for a second try (e.g., "poles" -> "pole")
    if s.endswith("s") and len(s) > 1:
        s_singular = s[:-1]
        if s_singular in ALIASES:
            return ALIASES[s_singular]
        if s_singular in CANONICAL:
            return s_singular

    # if exact canonical after cleaning, keep it
    if s in CANONICAL:
        return s

    # last chance: collapse spaces to compare (e.g., "waterbody")
    nospace = s.replace(" ", "")
    if nospace in ALIASES:
        return ALIASES[nospace]

    # Unknown: return cleaned token (won’t be counted as canonical unless GT uses same token)
    return s
