from utils.utils import DualLabelConverter
from utils.vn_tags import decompose_str, compose_str
from utils.utils import VIETNAMESE_BASE_CHARACTERS

base = VIETNAMESE_BASE_CHARACTERS
print('Base charset:', base)
conv = DualLabelConverter(base.values())
labels = ['Xin chào Việt Nam', 'điện thoại', 'ỳ ý y ỹ ỵ']
enc = conv.encode(labels)
print('encoded lens:', enc[1].tolist())
base_str, _, _ = decompose_str(labels[0])
print('base chars present:', set(base_str).issubset(set(base.values())))