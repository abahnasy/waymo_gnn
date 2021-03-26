import fire
import pickle


def calc_class_counts():
    classes = {'VEHICLE':0, 'PEDESTRIAN':0 , 'CYCLIST': 0 }
    with open("./data/Waymo/infos_train_01sweeps_filter_zero_gt.pkl", "rb") as f:
        waymo_infos = pickle.load(f)
        print("Using {} Frames".format(len(waymo_infos)))
        
        for info_file in waymo_infos:
            obj_list = info_file['gt_names']
            for obj in obj_list:
                if obj in classes.keys():
                    classes[obj] +=1
    
    total_count = sum([v for k, v in classes.items()])
    print(classes)
    print(total_count)
    classes_weights = {}
    for k, v in classes.items():
        weight = total_count/(3*classes[k])
        classes_weights[k] = weight
    print(classes_weights)
    

if __name__ == "__main__":
    fire.Fire()