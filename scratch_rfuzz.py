import rapidfuzz
import numpy as np

a = ["hello", "world"]
b = ["helo", "word"]

try:
    print("fuzz.ratio element-wise:")
    res = rapidfuzz.fuzz.ratio(a, b)
    print(res)
except Exception as e:
    print("Error with fuzz.ratio:", e)

try:
    print("cdist element-wise (queries=a, choices=b)?")
    res = rapidfuzz.process.cdist(a, b, scorer=rapidfuzz.fuzz.ratio)
    print(res)
    print("Diag:", np.diagonal(res))
except Exception as e:
    print("Error with cdist:", e)
