from resize_all import resize_all

RESIZED = True  # switch to False if images are not resized

if __name__ == '__main__':
    if not RESIZED:
        resize_all()
