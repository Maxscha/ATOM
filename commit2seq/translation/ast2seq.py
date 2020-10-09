from argparse import ArgumentParser
from config import Config
from model import Model

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
                        help="path to preprocessed dataset", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
                        help="path to test file", metavar="FILE", required=False)

    parser.add_argument("-s", "--save_prefix", dest="save_path_prefix",
                        help="path to save file", metavar="FILE", required=False)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument("-p", "--path_num", help="max path number", required=True)
    parser.add_argument("-e", "--embed", help="embedding dimension", default=128)
    parser.add_argument("-en", "--encoder", help="encoder dimension", default=256)
    parser.add_argument("-de", "--decoder", help="decoder dimension", default=256)
    parser.add_argument('--predict', action='store_true')
    args = parser.parse_args()

    config = Config.get_default_config(args)

    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        model.train()
    if config.TEST_PATH and not args.data_path:
        results, precision, recall, f1, current_bleu = model.evaluate()
        print('Accuracy: ' + str(results))
        print('Precision: ' + str(precision) + ', recall: ' + str(recall) + ', F1: ' + str(f1))
        print('Bleu: ' + str(current_bleu))
    model.close_session()
