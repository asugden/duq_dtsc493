# Imports at the top
import numpy as np
import pandas as pd
import scipy.optimize
from typing import Iterable, List, Tuple


# Two new lines before functions
def read(path: str) -> pd.DataFrame:
    """Read in the NYTimes covid data

    Args:
        path (str): location of nytimes data

    Returns:
        pd.DataFrame: raw NYTimes covid data
    
    """
    df = pd.read_csv(path)
    df['county'] = df['county'].str.lower()
    df['state'] = df['state'].str.lower()
    df['cases'] = df['cases'].diff()
    df['cases'] = df['cases'].fillna(0)
    return df


# Two lines before new functions, 
# except for class methods, where it's one line
def subset_county(df: pd.DataFrame, 
                  county: str, 
                  state: str, 
                  from_date: str,
                  fips: int = None) -> pd.DataFrame:
    """Subset the data to a specific county and time range

    Args:
        df (pd.DataFrame): raw NYTimes data
        county (str): county for subset, ignored if fips is set
        state (str): state for subset, ignored if fips is set
        from_date (str): first date to include
        fips (int, optional): county code, overrides county, state. 
            Defaults to None.

    Returns:
        pd.DataFrame: a subset of the orginal dataframe specific
            to a county and time range
    
    """
    if fips is None:
        df = df.loc[df['county'] == county.lower(), :]
        df = df.loc[df['state'] == state.lower(), :]
    else:
        df = df.loc[df['fips'] == fips, :]

    df['date'] = pd.to_datetime(df['date'])
    df = df.loc[df['date'] >= pd.to_datetime(from_date), :]

    return df.reset_index(drop=True)


def sir(s_init: int, 
        i_init: int, 
        r_init: int, 
        r0: float, 
        d: float, 
        n_days: int) -> List[Tuple[float, float, float]]:
    """SIR model that integrates over time

    Args:
        s_init (int): initial susceptible population
        i_init (int): initial infected population
        r_init (int): initial recovered population
        r0 (float): the r0 to use to propagate
            (number of people infected by one person)
        d (float): the d to use to propagate (days of infection)
        n_days (int): number of days to propagate

    Returns:
        List[Tuple[float, float, float]]: A list of tuples of S, I, R
            for each day
    """
    s, i, r, n, out = s_init, i_init, r_init, s_init + i_init + r_init, [(s_init, i_init, r_init)]
    gamma = 1.0/d
    beta = r0*gamma
    for _ in range(1, n_days + 1):
        s += -(beta*i*s)/n
        i += (beta*i*s)/n - gamma*i
        r += gamma*i
        out.append((s, i, r))
    return out


def fit_r0(cases: pd.Series, population: int = 1_250_578) -> Tuple[float]:
    """Fit R0 and optionally d

    Args:
        cases (pd.Series): the column called cases from NYTimes df
        population (int): the population of the county in question

    Returns:
        float, float: the r0 and d, respectively
    
    """
    i = cases.values[0]
    r = 0
    s = population - i - r
    def fit_function(x, r0) -> List[float]:
        return np.diff([v[1] for v in sir(s, i, r, r0, 10, len(x))])
    
    pars, _ = scipy.optimize.curve_fit(fit_function, np.arange(len(cases)-1), 
        cases[1:].values, p0=[1.1], bounds=[[0.5], [15]])
    return pars[0]


# SCRIPT SECTION ==========================
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
    allegheny = subset_county(df, 'allegheny', 'pennsylvania', '2021-12-26')
    print(allegheny)
    print(fit_r0(allegheny['cases']))