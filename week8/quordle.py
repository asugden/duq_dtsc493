import importlib.resources
from unittest import skip
import numpy as np
from typing import Union


wordle_answers = importlib.resources.read_text(
    'week8', 'wordle_answers.txt').split('\n')
wordle_guesses = set(importlib.resources.read_text(
    'week8', 'wordle_guesses.txt').split('\n') + wordle_answers)


def get_word() -> str:
    """Draw a random wordle word

    Returns:
        str: random 5-letter valid wordle word

    """
    return wordle_answers[np.random.randint(len(wordle_answers))]


def check_guess(word: str) -> bool:
    """Check whether a guess is valid

    Args:
        word (str): guess word

    Returns:
        bool: True if the guess is valid

    """
    if word in wordle_guesses:
        return True
    else:
        return False


def check_score(word: str, score: tuple[int], comparison: str) -> bool:
    """Check whether the word and score match a comparison

    Args:
        word (str): guess word
        score (tuple[int]): guess score
        comparison (str): real answer

    Returns:
        bool: True if comparison matches the score

    """
    unmatched_guess = []
    unmatched_comp = []

    for i, (guess_letter, real_letter, letter_score) in enumerate(zip(word, comparison, score)):
        if letter_score == 2:
            if guess_letter != real_letter:
                return False
        elif letter_score == 1:
            if guess_letter == real_letter:
                return False

            unmatched_guess.append(guess_letter)
        unmatched_comp.append(real_letter)

    for i, guess_letter in enumerate(unmatched_guess):
        if guess_letter in unmatched_comp:
            unmatched_comp.pop(unmatched_comp.index(guess_letter))
        else:
            return False

    return True


def matching_words(state: tuple[list[str], list[tuple[int]]]) -> int:
    """Return the number of words that match a score

    Args:
        state (tuple[list[str], list[tuple[int]]]): a list of words
            and associated scores

    Returns:
        int: count of words that match the score

    """
    steps = len(state[0])
    if steps == 0:
        return len(wordle_answers)

    count = 0
    for answer in wordle_answers:
        answer_count = 0
        for word, score in zip(state[0], state[1]):
            if check_score(word, score, answer):
                answer_count += 1
        if answer_count == steps:
            count += 1

    return count


class Wordle:
    def __init__(self):
        self.word = get_word()
        self.guesses = []
        self.scores = []
        self.word_length = 5

    def valid(self, word: str) -> bool:
        """Test whether a guess is valid

        Args:
            word (str): guess word

        Returns:
            bool: True if the guess is a valid guess

        """
        word = word.strip().lower()
        return len(word) == self.word_length and check_guess(word)

    def remaining(self) -> bool:
        """Test whether the count has available guesses left

        Returns:
            bool: True if guesses are remaining

        """
        return len(self.guesses) < 6

    def guess(self, word: str, skip_remaining: bool = False) -> Union[tuple[int], None]:
        """Make a wordle guess

        Args:
            word (str): guess word
            skip_remaining (bool): if True, skip a check on the remaining count
                (for Quordle or other alternate games)

        Returns:
            tuple[int] | None: 1 for a correct letter in the wrong place
                               2 for a correct letter in the correct place
                               0 for incorrect letter
                               or None for invalid guess

        """
        word = word.strip().lower()
        if not self.valid(word) and (not skip_remaining or self.remaining()):
            return None

        self.guesses.append(word)

        unmatched_guess = []
        unmatched_real = []
        out = [0]*self.word_length

        for i, (guess_letter, real_letter) in enumerate(zip(word, self.word)):
            if guess_letter == real_letter:
                out[i] = 2
            else:
                unmatched_guess.append(guess_letter)
                unmatched_real.append(real_letter)

        for i, guess_letter in enumerate(unmatched_guess):
            if guess_letter in unmatched_real:
                out[i] = 1
                unmatched_real.pop(unmatched_real.index(guess_letter))

        self.scores.append(tuple(out))
        return self.scores[-1]

    def state(self) -> tuple[list[str], list[tuple[int]]]:
        """Return the current game state with a list of words and scores

        Returns:
            tuple[list[str], list[tuple[int]]]: a list of words followed by a list of scores

        """
        return (self.guesses, self.scores)


class Quordle:
    def __init__(self):
        self.wordles = [Wordle() for i in range(4)]
        self.guesses = []

    def valid(self, word: str) -> bool:
        """Test whether a guess is valid

        Args:
            word (str): guess word

        Returns:
            bool: True if the guess is a valid guess

        """
        return self.wordles[0].valid(word)

    def remaining(self) -> bool:
        """Test whether the count has available guesses left

        Returns:
            bool: True if guesses are remaining

        """
        return len(self.guesses) < 10

    def guess(self, word: str) -> Union[tuple[tuple[int]], None]:
        """Make a Quordle guess

        Args:
            word (str): guess word

        Returns:
            tuple[tuple[int]] | None: 1 for a correct letter in the wrong place
                                      2 for a correct letter in the correct place
                                      0 for incorrect letter
                                      or None for invalid guess

        """
        if not self.valid(word) or not self.remaining:
            return None

        self.guesses.append(word)
        return tuple([wordle.guess(word, skip_remaining=True) for wordle in self.wordles])


if __name__ == '__main__':
    print(get_word())
    print(matching_words((['crane'], [(2, 2, 2, 0, 0)])))
