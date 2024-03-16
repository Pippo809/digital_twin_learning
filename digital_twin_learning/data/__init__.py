from .pointnet import PointNetDataset, DataModifier, RandomPCDChange
from .pointnet_sampling import PointNetDatasetSampling, PCDComplete, PCDCompletePoses

from torch.utils.data import DataLoader

from .config import HyperParam, CostConfig, ModelConfig


def get_dataloader(data_type='pointnet', batch_size= 1, **kwargs):
    """get the data-loader for specific data type
    
    args:
        data_type: dataset format
        batch_size: batch-size
        kwargs: data kwargs
    
    returns:
        pytorch dataloader
    """
    if data_type == 'pointnet':
        dataset = PointNetDataset(**kwargs)
    else:
        raise ValueError(f"data type {data_type} is not defined")
    
    dataloder = DataLoader(dataset, batch_size = batch_size, shuffle= True)

    return dataloder