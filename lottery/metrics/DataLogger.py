import csv
import os
from typing import Mapping, List

import pandas as pd


class DataLogger:
    def __init__(self, headers: List[str], file_name: str, base_path: str):
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.headers = headers

    def write(self, row: Mapping[str, any]):
        return self.write_rows([row])

    def write_rows(self, rows: List[Mapping[str, any]], map_headers=True):
        file_exists = os.path.exists(self.file_path)
        mode = "a" if file_exists else "w"
        with open(self.file_path, mode, newline="", encoding="UTF8") as fp:
            writer = csv.DictWriter(fp, fieldnames=self.headers)

            if not file_exists:
                writer.writeheader()

            mapped_rows = (
                [dict(zip(self.headers, row)) for row in rows] if map_headers else rows
            )
            writer.writerows(mapped_rows)

    def load(self):
        with open(self.file_path) as fp:
            return pd.read_csv(fp)
