import os

from detectron2.data.datasets.register_coco import register_coco_instances
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata

from .datasets.text import register_text_instances

# register plane reconstruction

_PREDEFINED_SPLITS_PIC = {
    "pic_person_train": ("pic/image/train", "pic/annotations/train_person.json"),
    "pic_person_val": ("pic/image/val", "pic/annotations/val_person.json"),
}

metadata_pic = {
    "thing_classes": ["person"]
}

_PREDEFINED_SPLITS_TEXT = {
    # training sets with polygon annotations
    #DPText_DETR
    "syntext1_poly_train_pos": ("syntext1/train_images", "syntext1/train_poly_pos.json"),
    "syntext2_poly_train_pos": ("syntext2/train_images", "syntext2/train_poly_pos.json"),
    "mlt_poly_train_pos": ("mlt/train_images","mlt/train_poly_pos.json"),
    "totaltext_poly_train_ori": ("totaltext/train_images_rotate", "totaltext/train_poly_ori.json"),
    "totaltext_poly_train_pos": ("totaltext/train_images_rotate", "totaltext/train_poly_pos.json"),
    "totaltext_poly_train_rotate_ori": ("totaltext/train_images_rotate", "totaltext/train_poly_rotate_ori.json"),
    "totaltext_poly_train_rotate_pos": ("totaltext/train_images_rotate", "totaltext/train_poly_rotate_pos.json"),
    "ctw1500_poly_train_rotate_pos": ("ctw1500/train_images_rotate", "ctw1500/train_poly_rotate_pos.json"),
    "lsvt_poly_train_pos": ("lsvt/train_images","lsvt/train_poly_pos.json"),
    "art_poly_train_pos": ("art/train_images_rotate","art/train_poly_pos.json"),
    "art_poly_train_rotate_pos": ("art/train_images_rotate","art/train_poly_rotate_pos.json"),
    #-------------------------------------------------------------------------------------------------------
    "totaltext_poly_test": ("totaltext/test_images_rotate", "totaltext/test_poly.json"),
    "totaltext_poly_test_rotate": ("totaltext/test_images_rotate", "totaltext/test_poly_rotate.json"),
    "ctw1500_poly_test": ("ctw1500/test_images","ctw1500/test_poly.json"),
    "art_test": ("art/test_images","art/test_poly.json"),
    "inversetext_test": ("inversetext/test_images","inversetext/test_poly.json"),
    #-------------------------------------------------------------------------------------------------------
    # TESTR
    "totaltext_poly_train": ("totaltext/train_images", "totaltext/train_poly.json"),
    "totaltext_poly_val": ("totaltext/test_images", "totaltext/test_poly.json"),
    "ctw1500_word_poly_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_poly.json"),
    "ctw1500_word_poly_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_poly.json"),
    "syntext1_poly_train": ("syntext1/images", "syntext1/annotations/train_poly.json"),
    "syntext2_poly_train": ("syntext2/images", "syntext2/annotations/train_poly.json"),
    "mltbezier_word_poly_train": ("mlt2017/images","mlt2017/annotations/train_poly.json"),
    "icdar2015_train": ("icdar2015/train_images", "icdar2015/train_poly.json"),
    "icdar2015_test": ("icdar2015/test_images", "icdar2015/test_poly.json"),
    "icdar2019_train": ("icdar2019/train_images", "icdar2019/train_poly.json"),
    "textocr_train": ("textocr/train_images", "textocr/annotations/train_poly.json"),
    "textocr_val": ("textocr/train_images", "textocr/annotations/val_poly.json"),
}

metadata_text = {
    "thing_classes": ["text"]
}


def register_all_coco(root="datasets"):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_PIC.items():
        # Assume pre-defined datasets live in `./datasets`.s
        register_coco_instances(
            key,
            metadata_pic,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_TEXT.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_text_instances(
            key,
            metadata_text,
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


register_all_coco()
