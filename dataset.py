def get_data_set():
    data = list()
    label = list()
    try:
        with open('Data/train-dev/train.txt', 'r', encoding="utf8") as f:
            lines = f.readlines()
        for line in lines:
            arr = line.split("\t")
            data.append(arr[0])
            label.append(arr[1])
    except Exception as ex:
        print(ex)
    return data, label
