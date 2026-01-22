from pathlib import Path

def collect_labels(*files):
    labels = set()
    for fp in files:
        for line in Path(fp).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                labels.add(parts[1])
    # 确保 O 在最前/存在
    labels.discard("O")
    labels = ["O"] + sorted(labels)
    return labels

print(collect_labels(
    "datasets/myner/train.txt",
    "datasets/myner/dev.txt",
    "datasets/myner/test.txt",
))
