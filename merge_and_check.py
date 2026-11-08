"""Merge the Kaggle train/valid/test folders into one flat images/ + labels/
folder (what prepare_data.py expects) and check the class order."""
import collections, os, re, shutil, sys

src, dst = sys.argv[1], sys.argv[2]
for sub in ("images", "labels"):
    os.makedirs(os.path.join(dst, sub), exist_ok=True)
for split in ("train", "valid", "test"):
    for sub in ("images", "labels"):
        d = os.path.join(src, split, sub)
        for name in os.listdir(d):
            shutil.copyfile(os.path.join(d, name), os.path.join(dst, sub, name))

print("images:", len(os.listdir(os.path.join(dst, "images"))),
      "labels:", len(os.listdir(os.path.join(dst, "labels"))))
ids = collections.defaultdict(collections.Counter)
for f in os.listdir(os.path.join(dst, "labels")):
    cat = re.match(r"[a-z]+", f.lower()).group()
    for line in open(os.path.join(dst, "labels", f)):
        if line.strip():
            ids[cat][line.split()[0]] += 1
for cat in sorted(ids):
    print(cat, ids[cat].most_common(3))
