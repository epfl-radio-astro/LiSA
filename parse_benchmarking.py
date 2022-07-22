import glob
import numpy as np
# 2148810 # 1025448
# 2149915

def str_to_MB(s):
    if 'MB' in s:
        s = float(s.replace('MB',''))/1000
        return s
    elif 'GB' in s:
        s = float(s.replace('GB',''))
        return s
    elif 'kB' in s:
        s = float(s.replace('kB',''))/1e6
        return s
    elif 'B' in s:
        s = float(s.replace('B',''))/1e9
        return s

title = "119x119x6668 domain"
outfiles =glob.glob("*-2149915_*.out") 
benchmarking = {}
for ofile in outfiles:
    tag = ''
    with open(ofile, "r") as o:
        for line in o:
            if line[0:3] != '>>>':
                continue
            if 'Profiling:' in line:
                benchmarking[tag].append(line)
            else:
                tag = line.split('   ')[-1].strip()
                if tag not in benchmarking:
                    benchmarking[tag] = []
benchmarking_summary = {}
for key in benchmarking:
    benchmarking_summary[key] = {}
    benchmarking_summary[key]['time'] = []
    benchmarking_summary[key]['RSS'] = []
    benchmarking_summary[key]['VMS'] = []
    benchmarking_summary[key]['SHR'] = []
    for perf in benchmarking[key]:
        rss, vms, shr, time  = [k.strip().split(' ')[-1] for k in perf.split('|')]
        print(key, rss, vms, shr, time)
        if 'min' in time:
            raw_time = float(time.replace('min',''))*60
        elif 'ms' in time:
            raw_time = float(time.replace('ms',''))/1000
        elif 's' in time:
            raw_time = float(time.replace('s',''))
        else:
            raise RuntimeError("Unsure how to parse time")
        benchmarking_summary[key]['time'].append(raw_time)
        benchmarking_summary[key]['RSS'].append(str_to_MB(rss))
        benchmarking_summary[key]['VMS'].append(str_to_MB(vms))
        benchmarking_summary[key]['SHR'].append(str_to_MB(shr))
keys = []
times = []
RSS, VMS, SHR = [], [], []
for key in benchmarking_summary:
    #RSS:       0B | VMS:       0B | SHR       0B | time: 
    t = benchmarking_summary[key]['time']

    times.append(sum(t)/len(t))
    RSS.append(sum(benchmarking_summary[key]['RSS'])/len(t))
    VMS.append(sum(benchmarking_summary[key]['VMS'])/len(t))
    SHR.append(sum(benchmarking_summary[key]['SHR'])/len(t))

    if 'MERGE' in key: key = 'MERGE'
    elif 'READ' in key: key = 'READ'
    elif 'DENOIS' in key: key = 'DENOISE'
    elif 'FIND' in key: key = 'FIND\nSOURCES'
    if 'AI' in key: key = 'CHARAC-\nTERIZE'
    keys.append(key)
    print(key, sum(t)/len(t))

import matplotlib.pyplot as plt
x = np.linspace(0,1, len(keys))
fig, ax = plt.subplots(figsize=(5,3))
plt.subplots_adjust(left=0.15, bottom=0.15, right=.85, top=.85, wspace=0, hspace=0)
twin1 = ax.twinx()
__,__,h1 = ax.hist(x, bins = len(keys), weights = times, color = 'skyblue', label = 'time')
__,__,h2 = twin1.hist(x, bins=len(keys), weights = RSS, histtype = 'step', color = 'red', label = 'RSS')
__,__,h3 = twin1.hist(x, bins=len(keys), weights = VMS, histtype = 'step', color = 'red', label = 'VMS', linestyle = 'dashed')
__,__,h4 = twin1.hist(x, bins=len(keys), weights = SHR, histtype = 'step', color = 'red', label = 'SHR', linestyle = 'dotted')

labels = [l[0].get_label() for l in [h1,h2,h3,h4]]
twin1.legend([h1[0],h2[0],h3[0],h4[0]],  labels, ncol=2)
ax.set_xticks( np.linspace(0.1,0.9, len(keys)), keys)
ax.set_ylabel('Time [s]')
ax.set_yscale('log')

twin1.set_ylabel("Memory [GB]")
twin1.set_yscale('log')
twin1.set_ylim([1e-3, 5e3])
ax.set_ylim([.001, 1e5])


ax.tick_params(axis='y', colors='dodgerblue')
twin1.tick_params(axis='y', colors='red')
ax.yaxis.label.set_color('dodgerblue')
twin1.yaxis.label.set_color('red')
plt.show()
plt.title(title)
plt.savefig("perf.pdf")


