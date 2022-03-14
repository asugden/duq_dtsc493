from bs4 import BeautifulSoup
import pandas as pd
import urllib.request


def scrape_website(path: str) -> str:
    """Return the string of the data from a website given a path

    Args:
        path (str): the location of the website

    Returns:
        str: return the contents of the website

    """
    website = urllib.request.urlopen(path)
    contents = website.read()
    website.close()

    soup = BeautifulSoup(contents)
    [s.extract() for s in soup(['[document]', 'title'])]
    visible_text = soup.getText().replace(
        '\r', '\n').replace('\t', '').lower().split('\n')
    visible_text = [t.strip() for t in visible_text if len(t.strip()) > 40]

    return visible_text


# Evaluation features
# Run evaluation on the first 5 results from google from the queries:
    # diversity statement site:duq.edu
    # covid aapi site:duq.edu
    # manually curated diversity page
# url length, number of slashes
# "asian" count
# "southeast asian" count
# "pacific island" count
# "aapi" count
# "organization" count
# counts for southeast asian countries
# zip code/fips code
# estimate the recruiting area
# estimate the demographics in the recruiting area


if __name__ == '__main__':
    url = 'https://www.duq.edu/make-a-gift-to-duquesne/giving-priorities/diversity-priorities'
    print(scrape_website(url))
