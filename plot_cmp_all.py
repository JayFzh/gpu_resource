import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib
del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams['font.sans-serif'] = ['Times New Roman']


RESULTS_ORIGINAL ='./tmp_original_balance_sim/'
RESULTS_SIMPLE_SVC ='./tmp_balance_sim/'
RESULTS_SIMPLE_AVC ='./tmp_balance_sim/'


sources=[RESULTS_ORIGINAL,RESULTS_ORIGINAL,RESULTS_SIMPLE_SVC,RESULTS_SIMPLE_SVC,RESULTS_ORIGINAL,RESULTS_SIMPLE_SVC,RESULTS_SIMPLE_SVC,RESULTS_ORIGINAL,RESULTS_ORIGINAL]
names=["Pensieve-AVC","MPC-AVC","BOLA-AVC","BFAST-AVC","Grad-HYBJ","BFAST-HYBJ","Grad-HYBP","LAAVS-HYBP","Grad-AVC"]
colors=["orange","blue","fuchsia","cyan","red","brown","mediumvioletred","dodgerblue","grey"]

#caculte: avg bitrate, avg rebuff time, variance,avg bitrate change,avg remaining time
labels=["QoE","Bitrate utility",
        "Bitrate"]
labels2=["Rebuffering penalty",
        "Rebuffering time","Smoothness penalty"]
labels3=["Switch times","Switch amptitude",
        "Standard deviation\nbitrate"]#"Quality\nstandard deviation"]
patterns = ('XXXXXXX','---','**', '++', '\\\\\\\\\\',  'oo','///', '||', 'OO', 'O', '.')

#step3
pensieve=[69.4,119.8,32.3,18.1, 111.1, 8.4,10.7, 1.0 ,0.85]
bola=[64.1,106.5,19.8,22.6,97.3,5.15,7.5,0.77,1.03]
fast_avc=[58.7,100.4,26.6,15.1,97.6,6.93,5.7,0.86,1.12]
fast_hybrid=[66.1,111.1,28.7,16.3,102.4,7.48,4.9,0.95,0.72]
mpc=[66.7,121.6,31.5,23.4,116.5,8.2,7.7,1.0,0.9]
laavs=[61.7,120.4,34.9,23.8,113.0,8.55,7.5,0.99,0.77]
hybrid1=[73.7,107.7,27.2,11.4,97.2,7.1,5.8,0.82,0.76]
hybrid=[80.7,110.95,21.75,8.5,100.84,5.66,4.16,0.79,0.7]
ba = [70.84,100.0,21.1,8.1,87.8,5.5,4.04,0.76,0.76]

pensieve_std=[99.0,44.2,43.1,9.1]
pensieve_std=[88.3,41.7,41.4,8.14,56.8,10.7,5.08,0.41,0.43]
bola_std=[88.6,37.2,32.4,20.5,47.4,8.43,3.71,0.22,0.5]
fast_avc_std=[93.6,43.7,38.6,11.8,50.5,10.05,3.07,0.45,0.67]
fast_hybrid_std=[99.2,49.7,42.3,12.2,62.0,11.0,3.33,0.62,0.5]
mpc_std=[100.7,44.0,33.7,25.7,56.1,8.77,3.87,0.36,0.47]
laavs_std=[106.7,46.9,42.5,16.9,60.3,11.7,4.87,0.52,0.47]
hybrid1_std=[86.1,38.7,36.7,7.5,44.9,9.55,2.89,0.27,0.37]
hybrid_std=[80.1,42.7,32.2,7.8,53.9,8.5,2.03,0.34,0.40]
ba_std = [78.15,41.1,33.48,6.23,51.67,8.72,1.84,0.35,0.42]

data=[pensieve,mpc,bola,fast_avc,hybrid,fast_hybrid,hybrid1,laavs,ba]


std=[pensieve_std,mpc_std,bola_std,fast_avc_std,hybrid_std,fast_hybrid_std,hybrid1_std,laavs_std,ba_std]
print(np.max(data,axis=0))
print(np.array(data))

order = np.array([4,2,8,0,3,1,5,6,7])
order2 = np.array([4,6,8,0,1,2,3,5,7])
colors = np.array(colors)[order]
names = np.array(names)[order2]
std = np.array(std)[order2]
data = np.array(data)[order2]

std1=np.array(std)[:,[0,1,4]]/np.array(hybrid)[[0,1,4]]#np.max(data,axis=0)
data1=np.array(data)[:,[0,1,4]]/np.array(hybrid)[[0,1,4]]#np.max(data,axis=0)

#plot fig
bar_width = 0.45
highlight_index= 0

plt.figure(figsize=(10.3,8))
plt.subplot(3,1,1)
plt.ylim(0,2.5)
plt.grid(linestyle="--",linewidth= 0.5)
to_legend=[]
for i in range(len(sources)):
    for j in range(len(labels)):
        plt.errorbar(j* 5 + bar_width *(i+0.5), data1[i][j], yerr=std1[i][j], fmt='-',ecolor="black",linewidth=0.8,capsize=2)
    l=plt.bar(np.arange(len(labels))*5+bar_width*(i+0.5),data1[i],label=names[i],ec=colors[i],color='',alpha=1,width=bar_width,hatch=patterns[i])
    if i==highlight_index+100:
        plt.bar(np.arange(len(labels)) * 5 + bar_width * (i + 0.5), data1[i], ec="brown", color='',
                alpha=1, width=bar_width,linewidth=1.5)
    else:
        plt.bar(np.arange(len(labels)) * 5 + bar_width*(i+0.5), data1[i], ec="black", color='',
            alpha=1, width=bar_width)
    to_legend.append(l)

plt.xticks(np.arange(len(data1[0])) * 5 + bar_width*4.5, labels,fontsize=21)
plt.yticks([0.5*i for i in range(5)],fontsize=21)
plt.arrow(13.15, 1.8, 0, 0.6,
             width=0.05,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.15,
              head_length=0.25,
             fc='red',
             ec='pink',label="better")
plt.text(13.5,2.1,"Better",fontsize=19,color="black")
plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                mode="expand", borderaxespad=0, ncol=5,frameon=False,fontsize=14.5)




std2=np.array(std)[:,[2,5,3]]/np.array(hybrid)[[2,5,3]]#np.max(data,axis=0)
data2=np.array(data)[:,[2,5,3]]/np.array(hybrid)[[2,5,3]]#np.max(data,axis=0)
plt.subplot(3,1,2)
plt.ylim(0,6)
plt.ylabel("Normalized value",fontsize=20)
plt.grid(linestyle="--",linewidth= 0.5)
to_legend=[]
for i in range(len(sources)):
    for j in range(len(labels2)):
        plt.errorbar(j* 5 + bar_width *(i+0.5), data2[i][j], yerr=std2[i][j], fmt='-',ecolor="black",linewidth=0.8,capsize=2)
    l=plt.bar(np.arange(len(labels2))*5+bar_width*(i+0.5),data2[i],label=names[i],ec=colors[i],color='',alpha=1,width=bar_width,hatch=patterns[i])
    plt.bar(np.arange(len(labels2)) * 5 + bar_width*(i+0.5), data2[i], ec="black", color='',
            alpha=1, width=bar_width)
    to_legend.append(l)

plt.xticks(np.arange(len(data2[0])) * 5 + bar_width*4.5, labels2,fontsize=21)

plt.yticks([1.0*i for i in range(7)],fontsize=21)
plt.arrow(13.15, 5.8, 0, -1.55,
             width=0.05,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.15,
              head_length=0.7,
             fc='red',
             ec='pink',label="better")
plt.text(13.5,5.1,"Better",fontsize=19,color="black")

std3=np.array(std)[:,[6,7,8]]/np.array(hybrid)[[6,7,8]]#np.max(data,axis=0)
data3=np.array(data)[:,[6,7,8]]/np.array(hybrid)[[6,7,8]]#np.max(data,axis=0)

plt.subplot(3,1,3)
plt.ylim(0,4)
plt.grid(linestyle="--",linewidth= 0.5)
to_legend=[]
for i in range(len(sources)):
    for j in range(len(labels3)):
        plt.errorbar(j* 5 + bar_width *(i+0.5), data3[i][j], yerr=std3[i][j], fmt='-',ecolor="black",linewidth=0.8,capsize=2)
    l=plt.bar(np.arange(len(labels3))*5+bar_width*(i+0.5),data3[i],label=names[i],ec=colors[i],color='',alpha=1,width=bar_width,hatch=patterns[i])
    plt.bar(np.arange(len(labels3)) * 5 + bar_width*(i+0.5), data3[i], ec="black", color='',
            alpha=1, width=bar_width)
    to_legend.append(l)

plt.xticks(np.arange(len(data3[0])) * 5 + bar_width*4.5, labels3,fontsize=21)
plt.yticks([1.0*i for i in range(5)],fontsize=21)

plt.arrow(13.15, 3.8, 0, -1.0,
             width=0.05,
             length_includes_head=True, # 增加的长度包含箭头部分
              head_width=0.15,
              head_length=0.45,
             fc='red',
             ec='pink',label="better")
plt.text(13.5,3.3,"Better",fontsize=19,color="black")



plt.tight_layout()
plt.savefig("./0726/compress_all_cmp0730.pdf",bbox_inches = 'tight')#"sim_originalAndSVC.png")

plt.show()