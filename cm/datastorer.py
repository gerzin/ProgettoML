#!/usr/bin/env python3
from datetime import datetime
import pandas as pd


class DataStorer:
    def __init__(self, filename=None):
        """
        Store the data in a pandas dataframe.
        The columns are created after the first push.
        Subsequent pushes must have the same signature

        Params:
            filename    --  name of the file where to save the data.
                            if no name is provided the data will be saved on a file
                            named with a timestamp
        """
        self.df = None
        self.filename = filename

    def save_to_file(self, filename=None):
        """save the data on a file in csv format.

        Params:
            filename    -- name of the file. Will override previous settings.
        """
        if filename is None and self.filename is None:
            self.filename = (datetime.now().strftime(
                "%d-%b-%Y (%H:%M:%S.%f)")) + ".data"
        name = filename if filename is not None else self.filename
        self.df.to_csv(name, index=False)

    def push(self, **kwargs):
        """
        Inserts the keyword arguments in the DataStorer
        params:
            kwargs  -- keyword arguments
                       (you must be consistent and call it with the same keywords used in the first push)
        """
        if self.df is None:
            names = []
            values = []
            for (k, v) in kwargs.items():
                names.append(k)
                values.append(v)
            self.df = pd.DataFrame(columns=(names))
        self.df = self.df.append(kwargs, ignore_index=True)


if __name__ == '__main__':
    # test
    a = DataStorer()
    a.push(name="Jane", surname="Doe", age=40, time=datetime.now())
    a.push(name="John", surname="Doe", age=42, time=datetime.now())
    a.save_to_file()
