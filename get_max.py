import pickle
import os


def get_max(filename):
    with open(filename, 'rb') as file:
        a = pickle.load(file)
        max_test_acc = max_train_acc = 0
        min_test_loss = min_train_loss = 9999
        for k in a:
            # print(f'Checking config {k}')
            if a[k].max_test_acc > max_test_acc:
                res = a[k]
                res_k = k
                print(f'Updating max_acc from {max_test_acc} to {a[k].max_test_acc} with config:\n{k}')
                max_test_acc = a[k].max_test_acc

            # max_test_acc   = max(max_test_acc, a[k]['max_test_acc'])
            # min_test_loss  = min(min_test_loss, a[k]['min_test_loss'])
            # max_train_acc  = max(max_train_acc, a[k]['max_train_acc'])
            # min_train_loss = min(min_train_loss, a[k]['min_train_loss'])

        print(f'Results for {filename}:\n'
              f'max_test_acc: {max_test_acc} | min_test_loss: {res.min_test_loss} | max_train_acc: {res.max_train_acc} | min_train_loss: {res.min_train_loss}')



def main():
    for filename in os.listdir('.'):
        if 'output' in filename and filename.endswith('.tmp'):
            get_max(filename)


if __name__ == '__main__':
    main()
