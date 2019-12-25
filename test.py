"""
-*- coding:utf-8 -*-
@Time   :2019/12/23 下午7:45
@Author :wts
@File   :test.py
@Version：1.0
"""
cpdomains = ["900 google.mail.com", "50 yahoo.com", "1 intel.mail.com", "5 wiki.org"]
dicts = {}
for i in cpdomains:
    t = ""
    st = ""
    for j in i:
        if(j != ' '):
            t += j
        else:
            break
    print(len(t))
    print(len(i))
    for k in range(len(i)-1,len(t)-1,-1):
        print(i[k])
        print(k)
        if(i[k] != '.'):
            if(i[k] == ' '):
                if (dicts.get(st) is None):
                    dicts[st] = int(t)
                else:
                    dicts[st] += int(t)
                break
            st += i[k]
        else:
            if(dicts.get(st) is None):
                dicts[st] = int(t)
            else:
                dicts[st] += int(t)
            st += i[k]

rs = []
for k,v in dicts.items():
    rs.append(str(v) + " " + k[::-1])
print(rs)
print(range(2,4)[0])
