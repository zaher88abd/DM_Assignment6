class DataSet:
    def get_test_data_set(file_name='Data/test/test.txt'):
        with open(file_name, 'r', encoding="utf8")as f:
            data = f.readlines()
        return data

    def get_train_data_set(file_name='Data/train-dev/train.txt'):
        data = list()
        label = list()
        try:
            with open(file_name, 'r', encoding="utf8") as f:
                lines = f.readlines()
            for line in lines:
                arr = line.split("\t")
                data.append(arr[0])
                label.append(arr[1])
        except Exception as ex:
            print(ex)
        return data, label
