from resize_all import resize_all
from tools import init_paths, RAW_DATA_PATH

RESIZED = False
if __name__ == '__main__':
    init_paths(RAW_DATA_PATH)
    if not RESIZED:
        resize_all(RAW_TEST_PATH, PROCESSED_TEST_PATH, labels=False)
