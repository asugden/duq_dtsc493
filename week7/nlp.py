import fasttext
import numpy as np


def tweet_meaning(tweet: str) -> np.ndarray:
    words = tweet.split(' ')
    # Convert to lowercase
    words = [word.lower() for word in words]
    # Get vector
    embeddings = [model.get_word_vector(word) for word in words]
    return np.average(embeddings, axis=0)


model = fasttext.load_model('data/cc.en.50.bin')

if __name__ == '__main__':
    # print(model.get_word_vector('king'), model.get_word_vector('queen'))
    model.get_nearest_neighbors('hamster')
    print(model.get_nearest_neighbors('hamster'))
    nn = tweet_meaning('The buck stops here')

    print(model.get_nearest_neighbors(nn))
