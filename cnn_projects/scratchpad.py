import h5py


def h5_playground():
    f = h5py.File('datasets/train_happy.h5', "r")
    '''
    print(f.keys())
    < KeysViewHDF5['list_classes', 'train_set_x', 'train_set_y'] >
    '''

    '''
        (600, 64, 64, 3)
        600: images
        64: width
        64: height
        3: RGB
    '''
    train_set_X = f['train_set_x']

    train_set_y = f['train_set_y']
    classes = f['list_classes']

    print(train_set_X.shape)
    print(train_set_y.shape)
    print(classes.shape)

    from PIL import Image

    '''
    PIL Image requires numpy array with uint8 type in H, W, C format
    '''
    img = Image.fromarray(train_set_X[1])
    img.show()
