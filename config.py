import warnings


class DefaultConfig(object):
    env = 'default'
    model = 'CNN_2d'

    train_data_root = './data/train/Pat1_small/60sec'
    test_data_root = './data/test/60sec/'
    load_model_path = './cnn_2d_small_60.pth'

    batch_size = 128
    num_workers = 4
    print_freq = 20

    debug_file = '/tmp/debug'
    result_file = 'result_small_60sec.csv'

    max_epoch = 100
    lr = 0.1
    lr_decay = 0.95
    weight_decay = 1e-4

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))


opt = DefaultConfig()
