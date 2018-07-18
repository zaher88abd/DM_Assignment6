class DataSet:
    def __init__(self):
        self.tranning.data = None
        self.tranning.label = None
        self.testing.data = None

    def get_test_data_set(self, file_name='Data/test/test.txt'):
        if self.testing.data is not None:
            return self.testing.data
        with open(file_name, 'r', encoding="utf8")as f:
            self.testing.data = f.readlines()
        return self.testing.data

    def get_train_data_set(self, file_name='Data/train-dev/train.txt'):
        if self.tranning.data is not None:
            return self.tranning.data, self.tranning.label
        else:
            self.tranning.data = list()
            self.tranning.label = list()
            try:
                with open(file_name, 'r', encoding="utf8") as f:
                    lines = f.readlines()
                for line in lines:
                    arr = line.split("\t")
                    self.tranning.data.append(arr[0])
                    self.tranning.data.append(arr[1])
            except Exception as ex:
                print(ex)
            return self.tranning.data, self.tranning.label
