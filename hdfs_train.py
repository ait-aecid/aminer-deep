import random

random.seed(1)

with open('data/hdfs_xu/sorted.log') as f, open('data/hdfs_xu/train.log', 'w+') as out, open('anomaly_label.csv') as label:
    anom = []
    norm = []
    header = True
    for line in label:
        if header:
            header = False
            continue
        parts = line.split(',')
        if parts[1].startswith("Anomaly"):
            anom.append(parts[0])
        else:
            norm.append(parts[0])
    print(len(norm))
    norm_selected = random.sample(list(norm), 5582) #[:5582]
    anom = set(anom)
    norm = set(norm)
    norm_selected = set(norm_selected)
    i = 0
    for line in f:
        i += 1
        if i % 500000 == 0:
            print(i)
        parts = line.split(' ')
        blkfound = False
        for part in parts:
            part = part.strip("\n\r.")
            if part.startswith('blk_'):
                blkfound = True
                if part in anom:
                    #print('a')
                    pass
                elif part in norm:
                    #print('b')
                    if part in norm_selected:
                        out.write(line)
                        break
                else:
                    print(part)
        if blkfound == False:
            print(line)
