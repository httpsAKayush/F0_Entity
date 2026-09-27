import rapidfuzz
a = ["hello", "world"]
b = ["helo", "word"]
try:
    print(rapidfuzz.process.cpdist(a, b, scorer=rapidfuzz.fuzz.ratio))
except Exception as e:
    print(e)
