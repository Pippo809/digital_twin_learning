from .architectures import KPFCNN
from .KPConfig import KPConfig


class KPinit():
    def __init__(self, k, emb_dim, dropout, output_channels=40, pose_features=False, use_normals=False):
        self.model = createKPmodel()


def createKPmodel():
    config = KPConfig()
    net = KPFCNN(config, [0.0, 0.3, 0.6, 1.0], [])
    return net
