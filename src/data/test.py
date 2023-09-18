import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import time

import pandas as pd
from requests_html import HTMLSession

# https://github.com/psf/requests-html/issues/275#issuecomment-513992564
session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)
URL = "https://www.immoweb.be/en/search/house/for-rent?countries=BE&page=1&orderBy=relevance"


def get_rendered_html(url):
    """
    Fetches the HTML content of a web page using a headless browser,
    renders it, and returns a specific element.

    Args:
        url (str): The URL of the web page to fetch and render.

    Returns:
        element: The first HTML element found matching the XPath expression
        '//*[@id="searchResults"]/div[4]/div/div[1]/div[1]/div[1]' on the rendered page.

    Note:
        This function uses a session object to fetch the
        web page content and a headless browser to render
        the JavaScript components, allowing for dynamic content retrieval.

    Example:
        To retrieve a specific element from a web page:

        >>> result_element = get_rendered_html("https://example.com")
        >>> if result_element:
        ...     # Process the retrieved HTML element
        ...     print(result_element.text)
        ... else:
        ...     print("Element not found on the page.")
    """
    r = session.get(url)
    r.html.render(sleep=1)
    return r.html.xpath(
        '//*[@id="searchResults"]/div[4]/div/div[1]/div[1]/div[1]', first=True
    )


def parse(links):
    """
    Parses data from a list of web page links, retrieves and processes HTML tables from each page.

    Args:
        links (set): A set of absolute URLs pointing to web pages containing HTML tables.

    Returns:
        list of DataFrame: A list of DataFrames, each containing a processed HTML table from a web page.

    Note:
        This function iterates through the provided set of links, fetches the content of each page,
        extracts HTML tables, and performs data cleaning. It adds metadata like the retrieval timestamp
        and the URL of the source page to each DataFrame.

    Example:
        To parse tables from a set of web page links:

        >>> links = {
        ...     'https://example.com/page1',
        ...     'https://example.com/page2',
        ...     'https://example.com/page3',
        ... }
        >>> parsed_data = parse(links)
        >>> for df in parsed_data:
        ...     # Process each DataFrame as needed
        ...     print(df.head())
    """
    all_tables_from_given_page = []
    for item in list(links.absolute_links):
        try:
            r = session.get(item)

            tables = pd.concat(pd.read_html(r.text)).dropna().set_index(0)
            tables.loc["day_of_retrieval", 1] = pd.Timestamp.now()
            tables.loc["ad_url", 1] = item

            all_tables_from_given_page.append(tables)
        except:
            pass
    dfs = [
        df.rename(columns={1: f"source_{i}"})
        for i, df in enumerate(all_tables_from_given_page)
    ]
    return dfs


def join_tables_on_same_page(dfs):
    """
    Joins a list of DataFrames horizontally and transposes the result.

    Args:
        dfs (list of DataFrame): A list of DataFrames to be horizontally joined.

    Returns:
        DataFrame: A DataFrame obtained by joining the input DataFrames
        horizontally and then transposing it.

    Example:
        To join and transpose a list of DataFrames:

        >>> df1 = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> df2 = pd.DataFrame({'C': [7, 8, 9], 'D': [10, 11, 12]})
        >>> df_list = [df1, df2]
        >>> result_df = join_tables_on_same_page(df_list)
        >>> print(result_df)

           0  1
        A  1  2
        B  4  5
        C  7  8
        D 10 11
    """
    return dfs[0].join(dfs[1:]).transpose()


def run_full_pipeline(kind_of_apartment):
    """
    Run a data scraping and processing pipeline for real estate listings on Immoweb.

    Args:
        kind_of_apartment (str): Specifies the type of apartment, either 'for_rent' or 'for_sale'.

    Returns:
        DataFrame: A DataFrame containing the complete dataset from the scraped and processed data.

    Note:
        This function scrapes real estate listings from Immoweb, including details about houses or apartments,
        and saves the data to CSV files. It iterates through multiple pages, fetches data, joins tables,
        and creates a complete dataset. The `kind_of_apartment` parameter determines whether to fetch
        listings for rent or sale.

    Example:
        To scrape and process real estate listings for rent:

        >>> complete_data = run_full_pipeline('for_rent')
        >>> print(complete_data.head())

           Property Type   Price (EUR)  Bedrooms  Bathrooms  Area (m²)  ...
        0   House          1200         3         2          1500       ...
        1   Apartment      950          2         1          100        ...
        ...              ...          ...       ...        ...        ...

        This function will save intermediate and complete datasets to CSV files and print 'Task is completed!'
        when finished.
    """
    all_tables = []

    PAGE = 106

    while True:  # Adjust the range if you want to scrape multiple pages
        try:
            # Generate the URL for the current page
            url = f"https://www.immoweb.be/en/search/house/{kind_of_apartment}?countries=BE&page={PAGE}&orderBy=relevance"

            # Fetch and render the page, then extract links
            links = get_rendered_html(url)
            print(f"Getting links from page {PAGE}.")

            # Parse data from the retrieved links
            parsed_data = parse(links)

            # Join the parsed data into a single table
            joined_tables = join_tables_on_same_page(parsed_data)

            # Save the joined table to a CSV file
            joined_tables.to_csv(
                f"joined_tables_{str(pd.Timestamp.now())[:10]}_{PAGE}.csv", index=False
            )

            all_tables.append(joined_tables)

            PAGE += 1

            # Add a sleep duration to avoid overloading the server with requests
            time.sleep(2)
        except:
            print("No more items!")
            break

    complete_dataset = pd.concat(all_tables, axis=0)
    complete_dataset.to_csv(
        f"complete_dataset_{str(pd.Timestamp.now())[:10]}.csv", index=False
    )

    print("Task is completed!")
    return complete_dataset


def run_full_pipeline_for_loop(kind_of_apartment):
    """
    Run a data scraping and processing pipeline for real estate listings on Immoweb.

    Args:
        kind_of_apartment (str): Specifies the type of apartment, either 'for_rent' or 'for_sale'.

    Returns:
        DataFrame: A DataFrame containing the complete dataset from the scraped and processed data.

    Note:
        This function scrapes real estate listings from Immoweb, including details about houses or apartments,
        and saves the data to CSV files. It iterates through multiple pages, fetches data, joins tables,
        and creates a complete dataset. The `kind_of_apartment` parameter determines whether to fetch
        listings for rent or sale.

    Example:
        To scrape and process real estate listings for rent:

        >>> complete_data = run_full_pipeline('for_rent')
        >>> print(complete_data.head())

           Property Type   Price (EUR)  Bedrooms  Bathrooms  Area (m²)  ...
        0   House          1200         3         2          1500       ...
        1   Apartment      950          2         1          100        ...
        ...              ...          ...       ...        ...        ...

        This function will save intermediate and complete datasets to CSV files and print 'Task is completed!'
        when finished.
    """
    all_tables = []

    PAGE = 100

    for page in range(1, PAGE):  # Adjust the range if you want to scrape multiple pages
        # Generate the URL for the current page
        url = f"https://www.immoweb.be/en/search/house/{kind_of_apartment}?countries=BE&page={page}&orderBy=relevance"

        # Fetch and render the page, then extract links
        links = get_rendered_html(url)
        print(f"Getting links from page {page}.")

        # Parse data from the retrieved links
        parsed_data = parse(links)

        # Join the parsed data into a single table
        joined_tables = join_tables_on_same_page(parsed_data)

        # Save the joined table to a CSV file
        joined_tables.to_csv(
            f"joined_tables_{str(pd.Timestamp.now())[:10]}_{page}.csv", index=False
        )

        all_tables.append(joined_tables)

        # Add a sleep duration to avoid overloading the server with requests
        time.sleep(2)

    complete_dataset = pd.concat(all_tables, axis=0)
    complete_dataset.to_csv(
        f"complete_dataset_{str(pd.Timestamp.now())[:10]}.csv", index=False
    )

    print("Task is completed!")
    return complete_dataset


run_full_pipeline_for_loop(kind_of_apartment="for_rent")
