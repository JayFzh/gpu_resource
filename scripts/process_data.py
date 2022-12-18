import numpy as np
file = '/home/users/fzh/log/fitting_data/overlap2-debugged'
commout = './docs/comm'
compout = './docs/comp'
comp = []
comm = []
with open(file,'r') as f:
    while 1:
        temp = f.readline()
        if len(temp) == 0:
            break
        temp = temp.split('\t')
        if temp[0]<temp[1]:
            flag = False
            for item in comm:
                if item['size'] == temp[2] and item['thread'] == temp[4]:
                    item['time'].append(temp[0])
                    flag = True
            if flag == False:               
                CommtimeDict = {}
                CommtimeDict['time'] = [temp[0]]
                CommtimeDict['size'] = temp[2]
                CommtimeDict['thread'] = temp[4]
                comm.append(CommtimeDict)
        else:
            flag = False
            for item in comp:
                if item['size'] == temp[3] and item['thread'] == temp[4]:
                    item['time'].append(temp[1])
                    flag = True
            if flag == False:                         
                ComptimeDict = {}
                ComptimeDict['time'] = [temp[1]]
                ComptimeDict['size'] = temp[3]
                ComptimeDict['thread'] = temp[4]
                comp.append(ComptimeDict)
                
with open(commout,'w') as f:
    for item in comm:
        item['time'] = np.mean([float(i) for i in item['time']])
        item['size'] = float(item['size'])
        item['thread'] = int(item['thread'])
        print(item)
        f.write(str(item)+'\n')
with open(compout,'w') as f:
    for item in comp:
        item['time'] = np.mean([float(i) for i in item['time']])
        item['size'] = float(item['size'])
        item['thread'] = int(item['thread'])
        print(item)
        f.write(str(item)+'\n')