def print_dataframe(dataset):
    """输出一个DataFrame对象的相关信息"""
    print("--------data info--------")
    print(dataset.info)
    print("--------data dtypes--------")
    print(dataset.dtypes)
    print("--------data selected_dtypes--------")
    print(dataset.select_dtypes)
    print("--------data axes--------")
    print(dataset.axes)
    print("--------data shape--------")
    print(dataset.shape)
    print("--------data values--------")
    print(dataset.values)


def simply_print(dataset):
    """输出一个对象"""
    print("--------data--------")
    print(dataset)
