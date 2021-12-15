import numpy as np
import util
import sys

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
pred_path='output/p02X_pred.txt'

def main(train_path, valid_path, test_path, save_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on t-labels,
        2. on y-labels,
        3. on y-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        save_path: Path to save predictions.
    """
    pred_path_a = pred_path.replace(WILDCARD, 'a')
    pred_path_b = pred_path.replace(WILDCARD, 'b')
    pred_path_f = pred_path.replace(WILDCARD, 'f')

    # *** START CODE HERE ***
    #######################################################################################
    # Problem (a)
    x_train, t_train = util.load_dataset(train_path, label_col='t', add_intercept=True)
    x_test, t_test = util.load_dataset(test_path, label_col='t', add_intercept=True)

    model_t = LogisticRegression()
    model_t.fit(x_train, t_train)

    util.plot(x_test, t_test, model_t.theta, 'output/p02a.png')

    t_pred_a = model_t.predict(x_test)
    np.savetxt(pred_path_a, t_pred_a > 0.5, fmt='%d')
    #######################################################################################
    # Problem (b)
    x_train, y_train = util.load_dataset(train_path, label_col='y', add_intercept=True)
    x_test, y_test = util.load_dataset(test_path, label_col='y', add_intercept=True)

    model_y = LogisticRegression()
    model_y.fit(x_train, y_train)

    util.plot(x_test, y_test, model_y.theta, 'output/p02b.png')

    y_pred = model_y.predict(x_test)
    np.savetxt(pred_path_b, y_pred > 0.5, fmt='%d')
    #######################################################################################  
    # Problem (f)
    x_valid, y_valid = util.load_dataset(valid_path, label_col='y', add_intercept=True)

    alpha = np.mean(model_y.predict(x_valid))

    correction = 1 + np.log(2 / alpha - 1) / model_y.theta[0]
    util.plot(x_test, t_test, model_y.theta, 'output/p02f.png', correction)

    t_pred_f = y_pred / alpha
    np.savetxt(pred_path_f, t_pred_f > 0.5, fmt='%d')
    #######################################################################################
    # *** END CODER HERE
    
if __name__ == '__main__':
    main(train_path='train.csv',
        valid_path='valid.csv',
        test_path='test.csv',
        save_path='posonly_X_pred.txt')


