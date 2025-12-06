"""
Example command to run the script:
python src/utils/dino_struct_sim.py \
    --folder_input "${F_BASE}/input" --folder_edit "${F_BASE}/edit"  --folder_output "${F_BASE}/metrics"

VitExtractor reference:
    https://github.com/omerbt/Splice/blob/master/models/extractor.py

"""

from glob import glob
import pdb
import warnings
import numpy as np
from PIL import Image
import torch, torchvision
import torch.nn.functional as F
import os, sys, argparse
from tqdm import tqdm
import json
from collections import OrderedDict


def attn_cosine_sim(x, eps=1e-08):
    x = x[0]  # TEMP: getting rid of redundant dimension, TBF
    norm1 = x.norm(dim=2, keepdim=True)
    factor = torch.clamp(norm1 @ norm1.permute(0, 2, 1), min=eps)
    sim_matrix = (x @ x.permute(0, 2, 1)) / factor
    return sim_matrix


class VitExtractor:
    BLOCK_KEY = 'block'
    ATTN_KEY = 'attn'
    PATCH_IMD_KEY = 'patch_imd'
    QKV_KEY = 'qkv'
    KEY_LIST = [BLOCK_KEY, ATTN_KEY, PATCH_IMD_KEY, QKV_KEY]

    def __init__(self, model_name, device):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.hook_handlers = []
        self.layers_dict = {}
        self.outputs_dict = {}
        for key in VitExtractor.KEY_LIST:
            self.layers_dict[key] = []
            self.outputs_dict[key] = []
        self._init_hooks_data()

    def _init_hooks_data(self):
        self.layers_dict[VitExtractor.BLOCK_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.ATTN_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.QKV_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.layers_dict[VitExtractor.PATCH_IMD_KEY] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for key in VitExtractor.KEY_LIST:
            # self.layers_dict[key] = kwargs[key] if key in kwargs.keys() else []
            self.outputs_dict[key] = []

    def _register_hooks(self, **kwargs):
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in self.layers_dict[VitExtractor.BLOCK_KEY]:
                self.hook_handlers.append(block.register_forward_hook(self._get_block_hook()))
            if block_idx in self.layers_dict[VitExtractor.ATTN_KEY]:
                self.hook_handlers.append(block.attn.attn_drop.register_forward_hook(self._get_attn_hook()))
            if block_idx in self.layers_dict[VitExtractor.QKV_KEY]:
                self.hook_handlers.append(block.attn.qkv.register_forward_hook(self._get_qkv_hook()))
            if block_idx in self.layers_dict[VitExtractor.PATCH_IMD_KEY]:
                self.hook_handlers.append(block.attn.register_forward_hook(self._get_patch_imd_hook()))

    def _clear_hooks(self):
        for handler in self.hook_handlers:
            handler.remove()
        self.hook_handlers = []

    def _get_block_hook(self):
        def _get_block_output(model, input, output):
            self.outputs_dict[VitExtractor.BLOCK_KEY].append(output)

        return _get_block_output

    def _get_attn_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.ATTN_KEY].append(output)

        return _get_attn_output

    def _get_qkv_hook(self):
        def _get_qkv_output(model, inp, output):
            self.outputs_dict[VitExtractor.QKV_KEY].append(output)

        return _get_qkv_output

    # TODO: CHECK ATTN OUTPUT TUPLE
    def _get_patch_imd_hook(self):
        def _get_attn_output(model, inp, output):
            self.outputs_dict[VitExtractor.PATCH_IMD_KEY].append(output[0])

        return _get_attn_output

    def get_feature_from_input(self, input_img):  # List([B, N, D])
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.BLOCK_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_qkv_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.QKV_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_attn_feature_from_input(self, input_img):
        self._register_hooks()
        self.model(input_img)
        feature = self.outputs_dict[VitExtractor.ATTN_KEY]
        self._clear_hooks()
        self._init_hooks_data()
        return feature

    def get_patch_size(self):
        return 8 if "8" in self.model_name else 16

    def get_width_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return w // patch_size

    def get_height_patch_num(self, input_img_shape):
        b, c, h, w = input_img_shape
        patch_size = self.get_patch_size()
        return h // patch_size

    def get_patch_num(self, input_img_shape):
        patch_num = 1 + (self.get_height_patch_num(input_img_shape) * self.get_width_patch_num(input_img_shape))
        return patch_num

    def get_head_num(self):
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self):
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_queries_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        q = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[0]
        return q

    def get_keys_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        k = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[1]
        return k

    def get_values_from_qkv(self, qkv, input_img_shape):
        patch_num = self.get_patch_num(input_img_shape)
        head_num = self.get_head_num()
        embedding_dim = self.get_embedding_dim()
        v = qkv.reshape(patch_num, 3, head_num, embedding_dim // head_num).permute(1, 2, 0, 3)[2]
        return v

    def get_keys_from_input(self, input_img, layer_num):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img, layer_num):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        h, t, d = keys.shape
        concatenated_keys = keys.transpose(0, 1).reshape(t, h * d)
        ssim_map = attn_cosine_sim(concatenated_keys[None, None, ...])
        return ssim_map


class DinoStructureLoss:
    def __init__(self, ):
        # from extractor import VitExtractor
        # imagenet_norm = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # global_resize_transform = transforms.Resize(224, max_size=480)
        self.extractor = VitExtractor(model_name="dino_vitb8", device="cuda")
        # self.global_transform = transforms.Compose([
        #                             global_resize_transform,
        #                             imagenet_norm
        #                         ])
        self.preprocess = torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])


    def calculate_global_ssim_loss(self, outputs, inputs):
        # pdb.set_trace()
        loss = 0.0
        for a, b in zip(inputs, outputs):  # avoid memory limitations
            # a = self.global_transform(a)
            # b = self.global_transform(b)
            with torch.no_grad():
                target_keys_self_sim = self.extractor.get_keys_self_sim_from_input(a.unsqueeze(0), layer_num=11)
            keys_ssim = self.extractor.get_keys_self_sim_from_input(b.unsqueeze(0), layer_num=11)
            loss += F.mse_loss(keys_ssim, target_keys_self_sim)
        return loss


def build_parser(parser):
    parser.add_argument('--folder_input', type=str, default=None, help='folder containing input images')
    parser.add_argument('--folder_edit', type=str, default=None, help='folder containing edited images')
    parser.add_argument('--folder_output', type=str, default=None, help='folder to save output images')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resize_edit_to_input', action="store_true")
    return parser


if __name__=="__main__":
    parser = build_parser(argparse.ArgumentParser())
    args = parser.parse_args()
    os.makedirs(args.folder_output, exist_ok=True)
    outf = os.path.join(args.folder_output, 'dino_struct_dist.json')
    if os.path.exists(outf):
        print(f'{outf} already exists, skipping')
        exit()
    d_output = {}
    # set the random seeds
    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    DINO_LOSS = DinoStructureLoss()
    l_input_images = glob(os.path.join(args.folder_input, '*.png'))
    for input_image in tqdm(l_input_images[::1]):
        bname = os.path.basename(input_image)
        edit_image = os.path.join(args.folder_edit, bname)
        # pil_target = Image.open(input_image).convert("RGB").resize((512,512), Image.Resampling.LANCZOS)
        # pil_edit = Image.open(edit_image).convert("RGB").resize((512,512), Image.Resampling.LANCZOS)
        pil_target = Image.open(input_image).convert("RGB")
        if args.resize_edit_to_input:
            pil_edit = Image.open(edit_image).convert("RGB").resize(pil_target.size, Image.Resampling.LANCZOS)
        
        with torch.no_grad():
            a = DINO_LOSS.preprocess(pil_target).unsqueeze(0).cuda()
            b = DINO_LOSS.preprocess(pil_edit).unsqueeze(0).cuda()
            dino_ssim = DINO_LOSS.calculate_global_ssim_loss(a, b).item()
        d_output[bname] = dino_ssim
    # get the mean of all 
    d_output['mean'] = np.mean(list(d_output.values()))
    # Create an OrderedDict with "mean" key at the top
    ordered_dict = OrderedDict({"mean": d_output.pop("mean")})
    ordered_dict.update(d_output)
    with open(outf, 'w') as f:
        json.dump(ordered_dict, f, indent=4)