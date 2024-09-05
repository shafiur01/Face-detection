from ts.torch_handler.base_handler import BaseHandler
import torch
import cv2
import numpy as np
import os
import logging
from data import cfg_re50  
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from .models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class RetinaFaceHandler(BaseHandler):
    def initialize(self, context):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = RetinaFace(cfg=cfg_re50, phase='test')
        model_pt_path = context.manifest['model']['serializedFile']
        self.model = self.load_model(self.model, model_pt_path, self.device == torch.device('cpu'))
        self.model.eval()
        logger.info('Model loaded and ready for inference.')

    def load_model(self, model, pretrained_path, load_to_cpu):
        logger.info(f"Loading model from {pretrained_path}")
        if load_to_cpu:
            pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            model = model.to('cpu')
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            model = model.to(device)
        
        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self.remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self.remove_prefix(pretrained_dict, 'module.')
            
        self.check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        return model

    def remove_prefix(self, state_dict, prefix):
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    def check_keys(self, model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True

    def preprocess(self, data):
        image = data[0].get("data") or data[0].get("body")
        npimg = np.frombuffer(image, np.uint8)
        img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        img = np.float32(img)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        return img

    def inference(self, img):
        loc, conf, landms = self.model(img)
        return loc, conf, landms

    def postprocess(self, inference_output):
        loc, conf, landms = inference_output
        img_raw = self.context.data[0]['img_raw']
        im_height, im_width, _ = img_raw.shape
        scale = torch.Tensor([im_width, im_height, im_width, im_height]).to(self.device)

        priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
        priors = priorbox.forward().to(self.device)
        boxes = decode(loc.data.squeeze(0), priors.data, cfg_re50['variance'])
        boxes = boxes * scale / 1
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, cfg_re50['variance'])
        landms = landms.cpu().numpy()

        inds = np.where(scores > 0.02)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, 0.4)
        dets = dets[keep, :]
        landms = landms[keep]

        for b in dets:
            if b[4] < 0.6:
                continue
            text = "{:.4f}".format(b[4])
            b = list(map(int, b))
            cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cx = b[0]
            cy = b[1] + 12
            cv2.putText(img_raw, text, (cx, cy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
            cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
            cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
            cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
            cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
            cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

        output_dir = "/workspaces/models/code/detected"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, "output.jpg")
        cv2.imwrite(output_path, img_raw)

        return [{"output_image": output_path}]