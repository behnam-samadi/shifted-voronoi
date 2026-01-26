import numpy as np

total_time = 0
exp = "0.15"
with open("runtime_"+exp+".txt") as f:
    lines = f.readlines()
for line in lines:
    line = line.split("\n")[0]
    line = float(line)
    #print(line)
    total_time += line

print(total_time/len(lines))

