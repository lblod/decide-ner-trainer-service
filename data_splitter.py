import os
import fire
import numpy as np


def split_data_simple(file_path: str, output_folder_path: str = "./data/", train_factor: float = 0.6, dev_factor: float = 0.2, test_factor: float = 0.2) -> None:

    if (train_factor + dev_factor + test_factor) != 1:
        raise ValueError(
            "The sum of the train, dev and test factors must be equal to 1!")

    os.makedirs(output_folder_path, exist_ok=True)

    with open(file_path) as f:
        lines = f.readlines()

    ii = np.where(np.array(lines) == "\n")[0]

    train_set_last_index = ii[int(len(ii) * train_factor)-1]
    dev_set_last_index = ii[int(len(ii) * (train_factor+dev_factor))-1]

    with open(os.path.join(output_folder_path, 'train.txt'), 'w') as f:
        f.writelines(lines[1:train_set_last_index])

    with open(os.path.join(output_folder_path, 'dev.txt'), 'w') as f:
        f.writelines(lines[train_set_last_index+1:dev_set_last_index])

    with open(os.path.join(output_folder_path, 'test.txt'), 'w') as f:
        f.writelines(lines[dev_set_last_index+1:])


if __name__ == '__main__':
    fire.Fire(split_data_simple)
