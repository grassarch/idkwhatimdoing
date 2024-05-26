# custom loss
import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment
from datasets import IterableDataset

import albumentations
import numpy as np
import cv2

import os
from PIL import Image, ImageDraw
import multiprocessing as mp

import json
from datasets import Dataset
import logging 
import pickle
import transformers
from transformers import AutoImageProcessor
from transformers import AutoProcessor

from transformers import AutoModelForZeroShotObjectDetection

from transformers import TrainingArguments

# subclass trainer
from transformers import Trainer

from datasets.utils.logging import disable_progress_bar
#disable our map progress
disable_progress_bar()

import warnings

# Ignore all warnings
warnings.simplefilter(action='ignore')
# Redirect warnings to the logging system
logging.captureWarnings(True)

# Set logging level to suppress warnings
logging.getLogger().setLevel(logging.CRITICAL)

#set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_data():
    with open('./get_data_info/new_annot.json', 'r') as fp:
        coco_annot = json.load(fp)

    coco_annot = coco_annot['images']
    id = 0
    for i in range(len(coco_annot)):
        caption = []
        category = []
        is_crowd = []
        bbox = []
        area = []
        id_li = []
        for annots in coco_annot[i]['annotations']:
            caption.append(annots['caption'])
            category.append(annots['category_id'])
            is_crowd.append(annots['is_crowd'])
            area.append(annots['area'])
            bbox.append(annots['bbox'])
            id_li.append(id)
            id += 1
        coco_annot[i]['annotations'] = {'caption': caption, 'category': category, 'is_crowd': is_crowd,
                                    'bbox': bbox, 'area': area, 'id': id_li}

    # image: path, annotations: list[dict(caption: category_id: is_crowd: area:)]
    # image_id
    vlm_data = Dataset.from_list(coco_annot)

    with open('./get_data_info/new_label.json', 'r') as fp:
        id2label = json.load(fp)
    #insert no detection
    #id2label['126'] = "no detection"
    label2id = {v: k for k, v in id2label.items()}
    categories = list(label2id.keys())

    text_inputs =list(id2label.values())

    def insert_no_detect(examples):
        for i, annotations in enumerate(examples["annotations"]):
            examples["annotations"][i]['area'].append(1)
            examples["annotations"][i]['caption'].append('no detection')
            #Fill entire image for no prediction
            #set our box to a exteremely small value
            examples["annotations"][i]["bbox"].append([0,0,1,1])
            examples["annotations"][i]["category"].append(126)
            examples["annotations"][i]["id"].append(examples["annotations"][i]["id"][-1]+1)
            #shift our ids
            examples["annotations"][i]["id"] = [j+examples["image_id"][i] for j in examples["annotations"][i]["id"]]
            examples["annotations"][i]['is_crowd'].append(1)
        return examples

    #vlm_data = vlm_data.map(insert_no_detect, batched=True, batch_size=16)
    #cast to iterable
    vlm_data = vlm_data.to_iterable_dataset(num_shards=4)
    return vlm_data, id2label, label2id, categories, text_inputs


#fetch our data here
_, id2label, label2id, categories, text_inputs = get_data()

#Put out processors here for initialisation
# using image processor from detr
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50", device="cuda:0")

checkpoint = "google/owlvit-base-patch32"
processor = AutoProcessor.from_pretrained(checkpoint)

transform = albumentations.Compose(
    [
        albumentations.Resize(448, 448),
        albumentations.HorizontalFlip(p=0.5),
        albumentations.RandomBrightnessContrast(p=0.1),     
    ],
    bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"], clip=True, ),
)


def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

# transforming a batch
def transform_aug_ann(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    transformed_data = []
    for image, objects in zip(examples["image"], examples["annotations"]):
        image = cv2.imread(f'../novice/images/{image}')
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
        #use an owlvit processor to translate text, image data
        transformed_data.append(processor(text=text_inputs, images=image, return_tensors="pt"))
    return {"transformed_data":transformed_data}

# transforming a batch
def transform_aug_ann_labels(examples):
    image_ids = examples["image_id"]
    images, bboxes, area, categories = [], [], [], []
    for image, objects in zip(examples["image"], examples["annotations"]):
        image = cv2.imread(f'../novice/images/{image}')
        #applying modifications to our image
        out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        area.append(objects["area"])
        images.append(out["image"])
        bboxes.append(out["bboxes"])
        categories.append(out["category"])
    targets = [
        {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
        for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
    ]
    
    #Use detr to process our images
    return image_processor(images=images, annotations=targets, return_tensors="pt")

def collate_fn(batch):
    input_ids = torch.Tensor(np.array([item["input_ids"] for item in batch])).int()
    input_ids = input_ids.to(device)
    attention_mask = torch.Tensor(np.array([item["attention_mask"] for item in batch])).int()
    attention_mask = attention_mask.to(device)
    pixel_values = torch.Tensor(np.array([item["pixel_values"] for item in batch]))
    pixel_values = pixel_values.to(device)
    labels = []
    for item in batch:
        for (key, value) in item["labels"].items():
            item["labels"][key] = torch.Tensor(value).to(device)
        labels.append(item["labels"])
     
    batch = {}
    batch["input_ids"] = input_ids
    batch["attention_mask"] = attention_mask
    batch["pixel_values"] = pixel_values
    batch["labels"] = labels
    return batch


class CustomConcatDataset(IterableDataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
    def __iter__(self):
        # Iterate over examples from both datasets
        for example_1, example_2 in zip(self.dataset1, self.dataset2):
            dict_ = {}
            dict_["input_ids"] = example_1["transformed_data"]["input_ids"]
            dict_["attention_mask"] = example_1["transformed_data"]["attention_mask"]
            dict_["pixel_values"] = example_1["transformed_data"]["pixel_values"][0]
            dict_["labels"] = example_2["labels"]
            yield dict_
    def __len__(self):
        #manually specify len of dataset here, the proper way was to call .format('torch') but im too lazy to set it up properly for this concat class
        return 5000
# Using Detr-Loss calculation https://github.com/facebookresearch/detr/blob/main/models/matcher.py
# https://www.kaggle.com/code/bibhasmondal96/detr-from-scratch
class BoxUtils(object):
    @staticmethod
    def box_cxcywh_to_xyxy(x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def box_xyxy_to_cxcywh(x):
        x0, y0, x1, y1 = x.unbind(-1)
        b = [(x0 + x1) / 2, (y0 + y1) / 2,
             (x1 - x0), (y1 - y0)]
        return torch.stack(b, dim=-1)

    @staticmethod
    def rescale_bboxes(out_bbox, size):
        #scale our boxes by image dims
        img_h, img_w = size
        b = BoxUtils.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    @staticmethod
    def box_area(boxes):
        """
        Computes the area of a set of bounding boxes, which are specified by its
        (x1, y1, x2, y2) coordinates.
        Arguments:
            boxes (Tensor[N, 4]): boxes for which the area will be computed. They
                are expected to be in (x1, y1, x2, y2) format
        Returns:
            area (Tensor[N]): area for each box
        """
        return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
    @staticmethod
    # modified from torchvision to also return the union
    def box_iou(boxes1, boxes2):
        area1 = BoxUtils.box_area(boxes1)
        area2 = BoxUtils.box_area(boxes2)

        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        union = area1[:, None] + area2 - inter

        iou = inter / union
        return iou, union

    @staticmethod
    def generalized_box_iou(boxes1, boxes2):
        """
        Generalized IoU from https://giou.stanford.edu/
        The boxes should be in [x0, y0, x1, y1] format
        Returns a [N, M] pairwise matrix, where N = len(boxes1)
        and M = len(boxes2)
        """
        # degenerate boxes gives inf / nan results
        # so do an early check
        assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
        assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
        iou, union = BoxUtils.box_iou(boxes1, boxes2)

        lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
        rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)  # [N,M,2]
        area = wh[:, :, 0] * wh[:, :, 1]

        return iou - (area - union) / area

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        # print(outputs, self.combine_target_batches(targets))
        # print(outputs.shape, targets)
        logging.info(f"{outputs.keys()=}")
        bs, num_queries = outputs["logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["class_labels"] for v in targets])
        logging.info(f"forward - {tgt_ids}")
        tgt_ids = tgt_ids.int()
        logging.info(f"forward - {tgt_ids}")

        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        # Compute the giou cost betwen boxes
        cost_giou = -BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(out_bbox),
            BoxUtils.box_cxcywh_to_xyxy(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        logging.info(f"loss_labels - {outputs.keys()}")
        assert 'logits' in outputs
        src_logits = torch.abs(outputs['logits'])
        device = src_logits.device
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["class_labels"][J] for t, (_, J) in zip(targets, indices)]).to(torch.int64)
        #remember to subtract to ensure that we have 0 to x-1 as label for x classes
        target_classes = torch.full(src_logits.shape[:2], self.num_classes-1,
                                    dtype=torch.int64, device=device).to(torch.int64)
        target_classes[idx] = target_classes_o
        #Expand our tensors to match batch_sizen
        src_logits = src_logits.transpose(1, 2)
        #we set the last class to be the bg
        not_bg = (target_classes != self.num_classes-1).unsqueeze(1).expand(-1, self.num_classes, -1)
        bg = (target_classes == self.num_classes-1).unsqueeze(1).expand(-1, self.num_classes, -1)
        # Apply the mask to the tensor 
        pred_logits = (src_logits * not_bg.float()).transpose(1, 2)
        bg_logits = (src_logits * bg.float()).transpose(1, 2)

        #pred_logits = src_logits[:, target_classes != 126].t()
        #bg_logits = src_logits[:, target_classes == 126].t()
        #target_classes = target_classes[target_classes != 126] # we already padded our classes with background

        # Positive loss
        pos_targets = torch.nn.functional.one_hot(target_classes, self.num_classes)
        #print(pos_targets)
        neg_targets = torch.zeros(bg_logits.shape).to(bg_logits.device)
        #print(self.empty_weight.float().shape, pred_logits.float().shape,pos_targets.float().shape)
        # import pickle
        # with open("empty_weight.pkl", 'wb') as f:
        #     pickle.dump(self.empty_weight.float(), f)
        # with open("pred_log.pkl", 'wb') as f:
        #     pickle.dump(pred_logits.float(), f)
        # with open("pos_targets.pkl", 'wb') as f:
        #     pickle.dump(pos_targets.float(), f)
        # assert False
        pos_loss = torch.nn.BCELoss(reduction="none", weight=self.empty_weight.float())(torch.nn.functional.softmax(pred_logits.float()), pos_targets.float())
        neg_loss = torch.nn.BCELoss(reduction="none", weight=self.empty_weight.float())(torch.nn.functional.softmax(bg_logits.float()), neg_targets.float())

        pos_loss = (torch.pow(1 - torch.exp(-pos_loss), 2) * pos_loss).sum(dim=1).mean()
        neg_loss = (torch.pow(1 - torch.exp(-neg_loss), 2) * neg_loss).sum(dim=1).mean()
        
        losses = {'loss_ce': pos_loss+neg_loss}

        return losses
    
    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["class_labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err.cpu()}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(BoxUtils.generalized_box_iou(
            BoxUtils.box_cxcywh_to_xyxy(src_boxes),
            BoxUtils.box_cxcywh_to_xyxy(target_boxes))
        )
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        Runs the relevant loss function
        """
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        logging.info(f"{type(outputs)=}")
        logging.info(f"{type(targets)=}")
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["class_labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))
        return losses

def custom_loss(logits, labels):
    num_classes = len(text_inputs)
    matcher = HungarianMatcher(cost_class = 1, cost_bbox = 5, cost_giou = 2)
    weight_dict = {'loss_ce': 1, 'loss_bbox': 5, 'loss_giou': 2}
    losses = ['labels', 'boxes', 'cardinality']
    criterion = SetCriterion(num_classes, matcher=matcher, weight_dict=weight_dict, eos_coef=0.1, losses=losses)
    criterion.to(device)
    loss = criterion(logits, labels)
    return loss

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        inputs["input_ids"] = torch.flatten(inputs["input_ids"], start_dim=0, end_dim=1)
        inputs["attention_mask"] = torch.flatten(inputs["attention_mask"], start_dim=0, end_dim=1)
        outputs = model(**inputs, return_dict=True)
        loss = custom_loss(outputs, labels)
        loss_ce = loss['loss_ce'].cpu().item()
        loss_bbox = loss['loss_bbox'].cpu().item()
        loss_giou = loss['loss_giou'].cpu().item()
        cardinality_error = loss['cardinality_error'].cpu().item()
        # print(
        #     f"loss_ce={loss_ce:.2f}",
        #     f"loss_bbox={loss_bbox:.2f}",
        #     f"loss_giou={loss_giou:.2f}",
        #     f"cardinality_error={cardinality_error:.2f}",
        #     sep="\t")
        loss = sum(loss.values())[0] #add
        return (loss, outputs) if return_outputs else loss

def main():
    #We set the last class a nodetect so now our model should perform poorly on it
    torch.cuda.empty_cache()
    mp.set_start_method('spawn', force=True)
    print("Current device is:", device)
    
    vlm_data, id2label, label2id, categories, text_inputs = get_data()
    model = AutoModelForZeroShotObjectDetection.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )
    model.to(device)

    print("Model loaded!")
    
    model_output_dir = "owlvit-base-patch32_newversion"
    
    transform_1 = vlm_data.map(transform_aug_ann, batched=True, batch_size=16)
    transform_2 = vlm_data.map(transform_aug_ann_labels, batched=True, batch_size=16)
    train_dataset = CustomConcatDataset(transform_1, transform_2)
    
    training_args = TrainingArguments(
                                        output_dir=model_output_dir,
                                        per_device_train_batch_size=16,
                                        warmup_steps=50,
                                        num_train_epochs=50,
                                        fp16=True,
                                        save_steps=50,
                                        logging_steps=50,
                                        learning_rate=1e-5, #1e-6
                                        weight_decay=1e-4,
                                        save_total_limit=20,
                                        lr_scheduler_type='cosine_with_restarts',
                                        remove_unused_columns=False,
                                        dataloader_pin_memory=False,
                                        gradient_checkpointing=True,
                                        gradient_accumulation_steps=1,
                                        push_to_hub=False,
                                        report_to=["tensorboard"],
                                        dataloader_num_workers=2,
                                        gradient_checkpointing_kwargs={"use_reentrant": False},
                                        )
    trainer = CustomTrainer(
                            model=model,
                            args=training_args,
                            data_collator=collate_fn,
                            train_dataset=train_dataset,
                            tokenizer=processor,
                            )
    
    eval_dict = trainer.train()
    with open("owl_vit_eval.pkl", "wb") as f:
        pickle.dump(eval_dict, f)
if __name__ == "__main__":
    main()
