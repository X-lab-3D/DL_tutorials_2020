from sklearn.model_selection import train_test_split
import torch

# Load data
def loadWine(file):
    """
    :param file: path to wine file
    :param label: int beloging to type of wine
    :return: list of lists with float values of wine characteristics
    """
    wine = open(file).readlines()[1:]
    wine = [w.strip().split(';') for w in wine ]
    wine = [[float(e) for e in w] for w in wine]

    return wine

def makeWineSamples(classification = True):
    red = loadWine("winequality-red.csv")
    white = loadWine("winequality-white.csv")

    if classification:
        red = [r + [0] for r in red]
        white = [w + [1] for w in white]

    redwhite = red + white
    wines = [w[:-1] for w in redwhite]
    labels = [w[-1] for w in redwhite]

    wines_train, wines_test, labels_train, labels_test = train_test_split(wines, labels, train_size=0.6)
    wines_test, wines_eval, labels_test, labels_eval = train_test_split(wines_test, labels_test, train_size = 0.5)

    wines_train = torch.tensor(wines_train)
    wines_test = torch.tensor(wines_test)
    wines_eval = torch.tensor(wines_eval)

    labels_train = torch.tensor(labels_train)
    labels_test = torch.tensor(labels_test)
    labels_eval = torch.tensor(labels_eval)

    return wines_train, wines_test, labels_train, labels_test, wines_eval, labels_eval

if __name__ == '__main__':
    print(len(red))
    print(len(white))  # there is more data on white wine. For now we wont bother
    print(len(redwhite))
