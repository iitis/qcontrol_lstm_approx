import numpy as np

def get_data(train_set_size,
            test_set_size,
            model_dim):

    train_input = []
    train_target = []
    test_input = []
    test_target = []
    train_indices = np.arange(train_set_size)
    test_indices = np.arange(train_set_size, train_set_size + test_set_size)
    if len(list(set(train_indices).intersection(set(test_indices)))) != 0:
        print("Test set overlaps with the training set!")

    print("To jest NOWY kod, ale NOWE dane")

    for i in train_indices:
        input = np.asarray(np.load("training/dim_{}/NCP_data/idx_{}.npz".format(
            model_dim, i))['arr_0'])
        u = np.asarray(np.load("training/dim_{}/mtx/idx_{}.npz".format(model_dim, i))['arr_0'])
        superoperator = np.kron(u, u.conjugate())

        train_input.append(input)
        train_target.append(superoperator)

    for i in test_indices:
        input = np.asarray(np.load("training/dim_{}/NCP_data/idx_{}.npz".format(
            model_dim, i))['arr_0'])
        u = np.asarray(np.load("training/dim_{}/mtx/idx_{}.npz".format(model_dim, i))['arr_0'])
        superoperator = np.kron(u, u.conjugate())

        test_input.append(input)
        test_target.append(superoperator)

    train_input = np.array(train_input)
    train_target = np.array(train_target)
    test_input = np.array(test_input)
    test_target = np.array(test_target)

    return (train_input, train_target, test_input, test_target)










































