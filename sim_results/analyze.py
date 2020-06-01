import pandas
import numpy as np
from collections import defaultdict

f = open("desease_stats_vstar.txt", "r")

results = defaultdict(list)

for lnum, line in enumerate(f):
    if lnum == 0:
        continue
    items = line[:-1].split("\t")
    key, value = items[0], items[1]
    if key == "SD_RECOVERED":
        results[key].append(value == "True")
    else:
        results[key].append(value)

f.close()

keys = list(results.keys())
for key in keys:
    if key.isdigit() or "MD5" in key or "ALL_DATA" == key:
        print("Dropping key: ", key)
        del results[key]
    else:
        print(key, len(results[key]))
        if key == "SD_RECOVERED":
            results[key] = np.array(results[key],dtype=bool)
        else:
            results[key] = np.array(results[key],dtype=np.float64)

df = pandas.DataFrame(data=results).infer_objects()

print(df.head())
print(df.columns)
print(df.dtypes)
df.to_pickle("../data/stats_star.xz")
