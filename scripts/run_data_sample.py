import os
import sys
import pickle
sys.path.append("../")
from tracing.chrome_tracing_v2 import *
from scripts.run_data_generate import _get_fusion_groups
from model import cluster_model
import numpy as np
import  matplotlib.pyplot as plt

# global setting
data_path = "/home/users/fzh/log/"
gpu_num = 2
model_settings = { "VGG16": [2, 4, 8, 16, 32, 64],
            "MobileNetV2": [2, 4, 8, 16, 32, 64],
                    "ResNet50":[2,4,8,16,32,64]}
'''
                 "ResNet152": [2, 4, 8, 16, 32, 64],
                  "Xception": [2, 4, 8, 16, 32],
                  "VGG16": [2, 4, 8, 16, 32, 64]}
'''
thread_options = [[64, 1], [128, 1], [256, 1], [512, 1],
                  [512, 2], [512, 4], [512, 8]]
model_info_fmt = "info-{}-b{}"
fitting_data = "/home/users/fzh/log/fitting_data/overlap2-debugged"
comp_fitting_data = "/home/users/fzh/log/fitting_data/overlap-comp-debugged"
unoverlap_comm_data = "/home/users/fzh/log/fitting_data/unoverlap-comm"

# load the model data
def load_model(model_name, batch_size, prefix = ""):
    file_path = prefix + data_path + model_info_fmt.format(model_name, batch_size)
    with open(file_path, 'rb') as f:
        info = pickle.load(f)
    #check if info indexes are consecutive
    ids = list(info.keys())
    ids.sort()
    id_num = len(ids)
    #if ids[0] != 1 or (ids[-1] - ids[0]) != id_num:
    new_info = {}
    for new_id, id in enumerate(ids):
        new_info[new_id + 1] = info[id]
        new_info[new_id + 1][0] = cluster_model.cluster_comm_size(new_info[new_id + 1][0])
    info = new_info

    return info

# further fuse info
def _fuse_info(ids, fusion_info):
    #print(fusion_info)
    #print(ids)
    fused_info = np.array(fusion_info[ids[0]])
    for id in ids[1:]:
        fused_info = fused_info + fusion_info[id]
    return list(fused_info)


# save the extract data to designated format
# reminder: fusion/layer_info = [comm_size, comp_time]
def save_extract_data(overlap_pairs, fusion_info, thread_alloc, bandwidth = 60):
    with open(fitting_data, 'a') as f:
        for entry in overlap_pairs:
            comm_ids = entry["comm_ids"]
            comp_ids = entry["comp_ids"]
            comm_time = entry["comm_time"]
            comp_time = entry["comp_time"]
            comm_fused = _fuse_info(comm_ids, fusion_info)
            comp_fused = _fuse_info(comp_ids, fusion_info)
            # format: [comm_time, comp_time, grad size, comp size, thread allocation]
            entry_fmt = "{}\t{}\t{}\t{}\t{}\t{}\n"
            entry = entry_fmt.format(comm_time, comp_time, comm_fused[0], comp_fused[1], thread_alloc, bandwidth)
            f.write(entry)

def save_multi_comp_data(comp_pairs,fusion_info,thread_alloc, bandwidth = 60):
    with open(comp_fitting_data,'a') as f:
        for entry in comp_pairs:
            comm_ids = entry["comm_ids"]
            comp_ids = entry["comp_ids"]
            comp_time = entry["comp_time"]
            comp_fused = _fuse_info(comp_ids, fusion_info)
            entry_fmt = "{}\t{}\t{}\t{}\t{}\n"
            # time size comm_num threadnum bandwidth
            entry = entry_fmt.format(comp_time,comp_fused[0],len(comm_ids),thread_alloc*len(comm_ids),bandwidth)
            f.write(entry)
            
def save_unoverlap_data(entry):
    with open(unoverlap_comm_data, 'a') as f:
        # format: [comm_time, comp_time, grad size, comp time, thread allocation]
        entry_fmt = "{}\t{}\t{}\t{}\t{}\t{}\n"
        entry = entry_fmt.format(entry[0], entry[1], entry[2], entry[3], entry[4], 60)
        f.write(entry)

if __name__ == "__main__0":
    for name in model_settings.keys():
        for batch_size in model_settings[name]:
            model_info = load_model(name, batch_size)
            for thread_alloc in thread_options:
                thread_alloc = thread_alloc[0] * thread_alloc[1]
                log_file = data_path + "{}_b{}_t{}".format(name, batch_size, thread_alloc)
                iter_stats, _ = get_layer_wise_info(log_file)
                # extract overlap data
                for iter_id in iter_stats.keys():
                    # debug
                    iter_pids = list(iter_stats[iter_id].keys())
                    model_pids = list(model_info.keys())
                    iter_pids.sort()
                    model_pids.sort()
                    #print("debug1: ", iter_pids)
                    #print("debug2: ", model_pids)
                    #exit()
                    fusion_iter, fusion_info = tensor_fusion(iter_stats[iter_id], model_info)
                    print_timeline(fusion_iter, fusion_info)
                    overlap_pairs = extract_contention_data(fusion_iter)
                    #print(fusion_info)
                    save_extract_data(overlap_pairs, fusion_info, thread_alloc)
                    print(log_file, iter_id)
                    #exit()

if __name__ == "__main__":
    # # debug in one file
    # name = 'ResNet50'
    # batch_size = 64
    # thread_alloc = 1024
    # group_spec = [22,22,22,22,21,21,21,21,21,21]
    # gi = 8
    # model_info = load_model(name, batch_size)
    # log_file = data_path + "{}_b{}_t{}_g{}".format(name,batch_size,thread_alloc,gi)
    # iter_stats, _ = get_layer_wise_info(log_file)
    # # extract overlap data
    # for iter_id in iter_stats.keys():
    #     # debug
    #     iter_pids = list(iter_stats[iter_id].keys())
    #     model_pids = list(model_info.keys())
    #     iter_pids.sort()
    #     model_pids.sort()
    #     print("debug1: ", iter_pids)
    #     print("debug2: ", model_pids)
    #     #exit()
    #     fusion_iter, fusion_info = tensor_fusion(iter_stats[iter_id], model_info, group_spec=group_spec)
    #     print_timeline(fusion_iter, fusion_info)
    #     overlap_pairs,comp_pairs = extract_contention_data(fusion_iter)
    #     #print(fusion_info)
    #     save_extract_data(overlap_pairs, fusion_info, thread_alloc)
    #     save_multi_comp_data(comp_pairs,fusion_info,thread_alloc)
    #     unoverlap_comm_entry = extract_unoverlap_communication(fusion_iter, fusion_info, thread_alloc)
    #     save_unoverlap_data(unoverlap_comm_entry)
    #     print(log_file, iter_id)
                
    for name in model_settings.keys():
        for batch_size in model_settings[name]:
            model_info = load_model(name, batch_size)
            tensor_num = len(model_info)
            for thread_alloc in thread_options:
                thread_alloc = thread_alloc[0] * thread_alloc[1]
                group_specs = _get_fusion_groups(tensor_num)
                for gi, group_spec in enumerate(group_specs):
                    log_file = data_path + "{}_b{}_t{}_g{}".format(name, batch_size, thread_alloc, gi)
                    iter_stats, _ = get_layer_wise_info(log_file)
                    # extract overlap data
                    for iter_id in iter_stats.keys():
                        # debug
                        iter_pids = list(iter_stats[iter_id].keys())
                        model_pids = list(model_info.keys())
                        iter_pids.sort()
                        model_pids.sort()
                        print("debug1: ", iter_pids)
                        print("debug2: ", model_pids)
                        #exit()
                        fusion_iter, fusion_info = tensor_fusion(iter_stats[iter_id], model_info, group_spec=group_spec)
                        print_timeline(fusion_iter, fusion_info)
                        overlap_pairs,comp_pairs = extract_contention_data(fusion_iter)
                        #print(fusion_info)
                        save_extract_data(overlap_pairs, fusion_info, thread_alloc)
                        save_multi_comp_data(comp_pairs,fusion_info,thread_alloc)
                        unoverlap_comm_entry = extract_unoverlap_communication(fusion_iter, fusion_info, thread_alloc)
                        save_unoverlap_data(unoverlap_comm_entry)
                        print(log_file, iter_id)



# if __name__ == "__main__":
    
#     filename = '/home/users/fzh/log/MobileNetV2_debug_4stream_{}.json'
#     thread_list = ["32"]
#     group = [40,40,40,38]
    
#     data = [[[] for i in range(0,len(group)*len(thread_list))],[[] for i in range(0,len(group)*len(thread_list))]]
#     for index,num in enumerate(thread_list):
#         log_file = filename.format(num)
#         iter_stats, _ = get_layer_wise_info(log_file)            
#         model_info = load_model('MobileNetV2',64,num)
#         for iter_id in iter_stats.keys():
#             # debug
#             iter_pids = list(iter_stats[iter_id].keys())
#             iter_pids.sort()
#             # print("debug1: ", iter_pids)
#             fusion_iter, fusion_info = tensor_fusion(iter_stats[iter_id], model_info, group)
#             # fusion_info is calculated by model_info
#             print_timeline(fusion_iter, fusion_info,data,index,len(thread_list))
#             overlap_pairs = extract_contention_data(fusion_iter)
#             # save_extract_data(overlap_pairs, fusion_info, 512)
#             # unoverlap_comm_entry = extract_unoverlap_communication(fusion_iter, fusion_info, 512)
#             # save_unoverlap_data(unoverlap_comm_entry)
#             print(log_file, iter_id)
        
    
    # for i in range(0,2):
    #     for j in range(0,len(group)*len(thread_list)):
    #         data[i][j] = np.mean(data[i][j])
    # print(data)
    # X = np.arange(len(group)*len(thread_list))
    # plt.bar(X - 0.2, data[0], color = 'r', width = 0.4,label = "comp")
    # plt.bar(X + 0.2, data[1], color = 'g', width = 0.4,label = "comm")
    # for i in range(len(data[0])):
    #     plt.text((i-0.2), data[0][i]*1.01, float(round(data[0][i],2)), ha='center', va= 'bottom',fontsize=4)
    #     plt.text((i+0.2), data[1][i]*1.01, float(round(data[1][i],2)), ha='center', va= 'bottom',fontsize=4)
    # x_list = []
    # fusion = '{}_{}'
    # for i in range(len(group)):
    #     for j in range(len(thread_list)):
    #         x_list.append(fusion.format(i+1,thread_list[j]))
    # plt.xticks(np.linspace(0,len(group)*len(thread_list)-1,len(group)*len(thread_list)),x_list,fontsize=5)
    # plt.legend(loc='upper left')
    # plt.grid()
    # plt.savefig('picture/2.png',dpi=150,bbox_inches = 'tight') 
        

    