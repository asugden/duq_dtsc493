# CSV: comma-separated variables
import matplotlib.pyplot as plt
import pandas as pd
import scipy.optimize
from typing import Iterable


def read(path: str) -> pd.DataFrame:
    """Read in CSV file of NYTimes data from Allegheny County

    Args:
        path (str): location of the file

    Returns:
        pd.DataFrame: read-in file for processing
    
    """
    df = pd.read_csv(path)
    df = df.loc[df['county'] == 'Allegheny', :]
    df = df.loc[df['state'] == 'Pennsylvania', :]
    df['cases'] = df['cases'].diff()
    df['cases'] = df['cases'].fillna(0)
    return df.reset_index(drop=True)


def calc_recovered(df: pd.DataFrame, pop_county: int = 1_250_578) -> float:
    """Calculate the recovered fraction

    Args:
        df (pd.DataFrame): NYTimes database for a single county
        pop_county (int, optional): Population size for that county. Defaults to 1_250_578.

    Returns:
        float: fraction of recovered cases
    """
    return df['cases'].sum()/pop_county


def sir(s_init: int, i_init: int, r_init: int, r0: float, d: float, n_days: int) -> Iterable[float]:
    s, i, r, n, out = s_init, i_init, r_init, s_init + i_init + r_init, [(s_init, i_init, r_init)]
    gamma = 1.0/d
    beta = r0*gamma
    for _ in range(1, n_days + 1):
        s += -(beta*i*s)/n
        i += (beta*i*s)/n - gamma*i
        r += gamma*i
        out.append((s, i, r))
    return out


def fit_sir(cases: pd.Series):
    cases = cases.values
    n = 1_250_578
    i = cases[0]
    s = n - cases[0]
    r = 0
    n_days = len(cases) - 1

    def fit_function(days, r0, d):
        return [v[1] for v in sir(s, i, r, r0, d, n_days)]

    best, _ = scipy.optimize.curve_fit(fit_function, range(n_days), cases, p0=[2, 10], bounds=[(0.5, 4), (5, 10)])
    return best


if __name__ == '__main__':
    # Anything after this will only be run if this file is the one that is run
    # If imported, it will not be run.
    # Double-underscore/dunder methods are built in python

    # Backslash is used for control characters
    # \n is newline
    # \t is tab
    # Because \ is a separator and because it's a control character, we need to use
    # \\ for every backslash
    df = read('data/us-counties.txt')
    # print(df)
    df['cases'].plot()
    plt.show()
    fit_sir(df['cases'].iloc[200:270])

