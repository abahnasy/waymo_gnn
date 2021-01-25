from .waymo import WaymoDataset

dataset_factory = {
    "WAYMO": WaymoDataset
}


def get_dataset(dataset_name):
    return dataset_factory[dataset_name]
