def resolve_path(path):
    return ROOT + path


def load_data(data_path):
    data = pd.read_csv(resolve_path(data_path))
    data["DATE"] = pd.to_datetime(data[["YEAR", "MONTH", "DAY"]])
    data.set_index("DATE", inplace=True)
    data = data[["CLOSE"]]
    return data
