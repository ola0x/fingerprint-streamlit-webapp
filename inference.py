import os
import streamlit as st
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

@st.cache(persist=True)
def initialization():
    """Loads configuration and model for the prediction.
    
    Returns:
        cfg (detectron2.config.config.CfgNode): Configuration for the model.
        predictor (detectron2.engine.defaults.DefaultPredicto): Model to use.
            by the model.
        
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.DEVICE = "cpu"
    cfg.MODEL.WEIGHTS = "model_final.pth"  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

    predictor = DefaultPredictor(cfg)
    return cfg, predictor

@st.cache
def inference(predictor, img):
    return predictor(img)