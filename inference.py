# from __future__ import print_function
# import os
# import argparse
# import torch
# import logging  
# import torch.backends.cudnn as cudnn
# import numpy as np
# from data import cfg_re50
# from layers.functions.prior_box import PriorBox
# from utils.nms.py_cpu_nms import py_cpu_nms
# import cv2
# from models.retinaface import RetinaFace
# from utils.box_utils import decode, decode_landm
# import time

# # Configure logging
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)
# logging.basicConfig(level=logging.DEBUG)

# parser = argparse.ArgumentParser(description='Retinaface')
# parser.add_argument('-m', '--trained_model', default='/opt/ml/model/code/model/Resnet50_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='resnet50', help='Backbone network or resnet50')
# parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
# parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
# parser.add_argument('--top_k', default=5000, type=int, help='top_k')
# parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
# parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
# parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
# parser.add_argument('image_path', type=str, help='Path to the image file')
# args, unknown = parser.parse_known_args()


# def check_keys(model, pretrained_state_dict):
#     ckpt_keys = set(pretrained_state_dict.keys())
#     model_keys = set(model.state_dict().keys())
#     used_pretrained_keys = model_keys & ckpt_keys
#     unused_pretrained_keys = ckpt_keys - model_keys
#     missing_keys = model_keys - ckpt_keys
#     print('Missing keys:{}'.format(len(missing_keys)))
#     print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
#     print('Used keys:{}'.format(len(used_pretrained_keys)))
#     assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
#     return True


# def remove_prefix(state_dict, prefix):
#     ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
#     print('Removing prefix \'{}\''.format(prefix))
#     f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
#     return {f(key): value for key, value in state_dict.items()}


# def load_model(model, pretrained_path, load_to_cpu):
#     # Log where the model is being loaded from
#     logger.info(f"Attempting to load pretrained model from: {pretrained_path}")
    
#     if load_to_cpu:
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
#     else:
#         device = torch.cuda.current_device()
#         logger.info(f"Using CUDA device: {device}")
#         pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    
#     if "state_dict" in pretrained_dict.keys():
#         logger.info('Found state_dict in the model file.')
#         pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
#     else:
#         logger.info('No state_dict found, removing prefix from the whole model.')
#         pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        
#     check_keys(model, pretrained_dict)
#     model.load_state_dict(pretrained_dict, strict=False)
#     logger.info(f"Model successfully loaded from: {pretrained_path}")
#     return model

# if __name__ == '__main__':
#     logger.info('Starting RetinaFace inference...')
#     torch.set_grad_enabled(False)
#     cfg = None
#     if args.network == "mobile0.25":
#         cfg = cfg_mnet
#     elif args.network == "resnet50":
#         cfg = cfg_re50
    
#     # net and model
#     logger.info(f'Initializing network with configuration: {args.network}')
#     net = RetinaFace(cfg=cfg, phase='test')
#     net = load_model(net, args.trained_model, args.cpu)
#     net.eval()
#     logger.info('Finished loading model!')
#     logger.debug(net)
#     cudnn.benchmark = True
#     device = torch.device("cpu" if args.cpu else "cuda")
#     logger.info(f'Using device: {device}')
#     net = net.to(device)

#     resize = 1

#     # Ensure "detected" directory exists
#     output_dir = "./detected"
#     if not os.path.exists(output_dir):
#         print(f'Creating output directory: {output_dir}')
#         os.makedirs(output_dir)

#     # Process the image
#     image_path = args.image_path
#     print(f'Loading image from path: {image_path}')
#     img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

#     img = np.float32(img_raw)

#     im_height, im_width, _ = img.shape
#     scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
#     img -= (104, 117, 123)
#     img = img.transpose(2, 0, 1)
#     img = torch.from_numpy(img).unsqueeze(0)
#     img = img.to(device)
#     scale = scale.to(device)

#     tic = time.time()
#     loc, conf, landms = net(img)  # forward pass
#     print('Net forward time: {:.4f} seconds'.format(time.time() - tic))

#     priorbox = PriorBox(cfg, image_size=(im_height, im_width))
#     priors = priorbox.forward()
#     priors = priors.to(device)
#     prior_data = priors.data
#     boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
#     boxes = boxes * scale / resize
#     boxes = boxes.cpu().numpy()
#     scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
#     landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
#     scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                            img.shape[3], img.shape[2], img.shape[3], img.shape[2],
#                            img.shape[3], img.shape[2]])
#     scale1 = scale1.to(device)
#     landms = landms * scale1 / resize
#     landms = landms.cpu().numpy()

#     # Ignore low scores
#     inds = np.where(scores > args.confidence_threshold)[0]
#     boxes = boxes[inds]
#     landms = landms[inds]
#     scores = scores[inds]

#     # Keep top-K before NMS
#     order = scores.argsort()[::-1][:args.top_k]
#     boxes = boxes[order]
#     landms = landms[order]
#     scores = scores[order]

#     # Do NMS
#     dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
#     keep = py_cpu_nms(dets, args.nms_threshold)
#     dets = dets[keep, :]
#     landms = landms[keep]

#     # Keep top-K faster NMS
#     dets = dets[:args.keep_top_k, :]
#     landms = landms[:args.keep_top_k, :]

#     dets = np.concatenate((dets, landms), axis=1)

#     # Show image
#     if args.save_image:
#         print(f'Saving detection results to {output_dir}')
#         for b in dets:
#             if b[4] < args.vis_thres:
#                 continue
#             text = "{:.4f}".format(b[4])
#             b = list(map(int, b))
#             cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
#             cx = b[0]
#             cy = b[1] + 12
#             cv2.putText(img_raw, text, (cx, cy),
#                         cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

#             # Landmarks
#             cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
#             cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
#             cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
#             cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
#             cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

#         # Save image with the same name in the "detected" directory
#         base_name = os.path.basename(image_path)
#         output_path = os.path.join(output_dir, base_name)
#         cv2.imwrite(output_path, img_raw)
#         print(f"Image saved to {output_path}")





from __future__ import print_function
import os
import argparse
import torch
import logging  
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_re50  
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network: resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('image_path', type=str, help='/curve/test.jpg')
args = parser.parse_args()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('Removing prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    # Log where the model is being loaded from
    logger.info(f"Attempting to load pretrained model from: {pretrained_path}")
    
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
        model = model.to('cpu')
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model = model.to(device)
    
    if "state_dict" in pretrained_dict.keys():
        logger.info('Found state_dict in the model file.')
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        logger.info('No state_dict found, removing prefix from the whole model.')
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
        
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"Model successfully loaded from: {pretrained_path}")
    return model

def predict_fn(image_path, model):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)  # Ensure model is on the correct device

        # Load and preprocess the image
        logger.info(f"Loading image from path: {image_path}")
        img_raw = cv2.imread(image_path, cv2.IMREAD_COLOR)

        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(device)
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0).to(device)  # Ensure input data is on the correct device

        # Perform forward pass
        tic = time.time()
        loc, conf, landms = model(img)  # forward pass
        logger.info('Net forward time: {:.4f} seconds'.format(time.time() - tic))

        priorbox = PriorBox(cfg_re50, image_size=(im_height, im_width))
        priors = priorbox.forward().to(device)
        boxes = decode(loc.data.squeeze(0), priors.data, cfg_re50['variance'])
        boxes = boxes * scale / 1
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), priors.data, cfg_re50['variance'])
        landms = landms.cpu().numpy()

        # Filter and apply NMS
        inds = np.where(scores > args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]
        order = scores.argsort()[::-1][:args.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, args.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        # Keep top-K detections
        dets = np.concatenate((dets, landms), axis=1)

        # Save the image with detections
        if args.save_image:
            logger.info(f"Saving detection results to './detected'")
            for b in dets:
                if b[4] < args.vis_thres:
                    continue
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(img_raw, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                # Landmarks
                cv2.circle(img_raw, (b[5], b[6]), 1, (0, 0, 255), 4)
                cv2.circle(img_raw, (b[7], b[8]), 1, (0, 255, 255), 4)
                cv2.circle(img_raw, (b[9], b[10]), 1, (255, 0, 255), 4)
                cv2.circle(img_raw, (b[11], b[12]), 1, (0, 255, 0), 4)
                cv2.circle(img_raw, (b[13], b[14]), 1, (255, 0, 0), 4)

            # Save image with the same name in the "detected" directory
            base_name = os.path.basename(image_path)
            output_path = os.path.join("./detected", base_name)
            cv2.imwrite(output_path, img_raw)
            logger.info(f"Image saved to {output_path}")

        return dets

    except Exception as e:
        logger.error(f"Error during prediction: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    logger.info('Starting RetinaFace inference...')
    torch.set_grad_enabled(False)

    # Ensure "detected" directory exists
    output_dir = "./detected"
    if not os.path.exists(output_dir):
        logger.info(f'Creating output directory: {output_dir}')
        os.makedirs(output_dir)

    # Initialize network and load model
    cfg = cfg_re50
    logger.info(f'Initializing network with ResNet50 configuration.')
    net = RetinaFace(cfg=cfg, phase='test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    logger.info('Finished loading model!')
    logger.debug(net)
    cudnn.benchmark = True

    # Perform inference
    predict_fn(args.image_path, net)
