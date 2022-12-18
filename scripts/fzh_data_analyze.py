import numpy as np
import sys
k = 10
s = 100
c = [0,]
b = [0,]
r = [0,]
tc = [[-1 for j in range(4)] for i in range(k)]
# tc[i][0] is begin 
# tc[i][1] is end
# tc[i][2] is size
# tc[i][3] is threadnum
tb = [[-1 for j in range(3)] for i in range(k)]
# tb[i][0] is begin 
# tb[i][1] is end
# tb[i][2] is size
t = [[-1 for j in range(2)] for i in range(k)]
# t[i][0] is begin
# t[i][1] is end
threadMap = [[0,sys.maxsize,0]]
def g(b,r):
    pass
def p(c,r):
    pass
def q(c,r):
    pass
def fo(size,r,begin,end,tnow):
    if size==0 or tnow<begin or tnow>end:
        return 0
    return min(end-begin,g(size,r))

def calEndTime(size,tnow):
    end = []
    for threaditem in threadMap:
        overlap = fo(s,threaditem[2],threaditem[0],threaditem[1],tnow)
        s = s*(1-overlap/g(s,threaditem[2]))
        end.append(overlap)
        if s==0:
            break
        else: tnow = threaditem[1]
    return sum(end)


def reCalMap(tc):
    global threadMap
    threadMap = [[0,sys.maxsize,0]]
    for item in tc:
        if item[3] == -1:
            continue
        temp = [item[0],item[1],item[3]]
        threadMap = insertMap(temp,threadMap)

def insertMap(a,threadMap):
    new = []
    startIndex = 0
    endIndex = 0
    for key,value in enumerate(threadMap):
        if value[0]<a[0] and value[1]>=a[0]:
            startIndex = key
        if value[0]<a[1] and value[1]>=a[1]:
            endIndex = key
    print(startIndex,endIndex)
    for i in range(0,startIndex):
            new.append(threadMap[i])
    if startIndex == endIndex:
        t1 = []
        t1.append(threadMap[startIndex][0])
        t1.append(a[0])
        t1.append(threadMap[startIndex][2])
        new.append(t1)

        t2 = []
        t2.append(a[0])
        t2.append(a[1])
        t2.append(a[2]+threadMap[startIndex][2])
        new.append(t2)

        t3 = []
        t3.append(a[1])
        t3.append(threadMap[startIndex][1])
        t3.append(threadMap[startIndex][2])
        new.append(t3)
    else:
        t1 = []
        t1.append(threadMap[startIndex][0])
        t1.append(a[0])
        t1.append(threadMap[startIndex][2])
        new.append(t1)
        t2 = []
        t2.append(a[0])
        t2.append(threadMap[startIndex][1])
        t2.append(threadMap[startIndex][2]+a[2])
        new.append(t2)
        for i in range(startIndex+1,endIndex):
            threadMap[i][2] += a[2]
            new.append(threadMap[i])
        t3 = []
        t3.append(threadMap[endIndex][0])
        t3.append(a[1])
        t3.append(threadMap[endIndex][2]+a[2])
        new.append(t3)
        t4 = []
        t4.append(a[1])
        t4.append(threadMap[endIndex][1])
        t4.append(threadMap[endIndex][2])
        new.append(t4)
    for i in range(endIndex+1,len(threadMap)):
        new.append(threadMap[i])
    for item in new:
        if item[0] == item[1]:
            new.remove(item)
    return new



if __name__ == "__main__":
    print(threadMap)
    tb[0][1] = 0
    t[1][0] = 0
    t[0][1] = calEndTime(s,0)
    for i in range(1,k+2):
        # stage i begin time
        t[i][0] = tb[i-1][1]
        # cal i-1 comm end
        tc[i-1][0] = t[i][0]
        tc[i-1][2] = c[i-1]
        for j in range(1,i-1):
            if p(c[j],r[j])<t[i-1][1]:
                tc[j+1][1] = tc[j+1][0]+p(c[j],r[j])
            else:
                tc[j+1][1] = t[i-1][1]+q(c[j]*(1-(t[i-1][1]-t[j+1][0])/p(c[j],r[j])),r[j])
        reCalMap(tc)
        if i != k+1:
            tb[i][0] = t[i][0]
            tb[i][1] = calEndTime(b[i],tb[i][0])
            tb[i][2] = b[i]
        remain = sum([b[j] for j in range(i+1,k+1)])
        t[i][1] = calEndTime(remain,tb[i][1])       
