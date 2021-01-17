import warnings


class DefaultConfig(object):
    env = 'default'
    model = 'CNN_1d'

    train_data_root = './data/train/'
    test_data_root = './data/test/'
    load_model_path = None

    batch_size = 128
    num_workers = 4
    print_freq = 20

    debug_file = '/tmp/debug'
    result_file = 'result.csv'

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
