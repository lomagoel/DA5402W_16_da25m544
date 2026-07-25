# load labelmap from label_map.txt
label_map_path =  "./label_map.txt"
with open(label_map_path, "r") as f:
    # load csv as a dictionary
    label_map = {}
    for line in f:
        cat, idx = line.strip().split()
        label_map[cat] = int(idx)
