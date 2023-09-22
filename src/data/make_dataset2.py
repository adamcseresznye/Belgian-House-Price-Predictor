import re
import time
from io import StringIO
from pathlib import Path
from typing import List, Set

import pandas as pd
from requests_html import Element, HTMLSession
from tqdm import tqdm

from data import utils


class ImmowebScraper:
    """
    A class for scraping and processing real estate listings from Immoweb.

    Args:
        session (requests_html.HTMLSession): An HTMLSession object for making
            HTTP requests.

    Attributes:
        session (requests_html.HTMLSession): The HTMLSession object used for
            making HTTP requests.

    Methods:
        get_last_page_number_from_url(url: str) -> int:
            Retrieve the number of the last page from a given Immoweb search
            results page URL.

        get_links_to_listings(url: str) -> Element:
            Get the dynamic HTML content of a web page using a headless browser.

        extract_ads_from_given_page(links: Set[str]) -> List[pd.DataFrame]:
            Extract and process HTML tables from a list of web page links.

        join_and_transpose_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
            Joins a list of DataFrames horizontally and transposes the result.

        reformat_scraped_dataframe(df: pd.DataFrame, columns_to_keep: List[str]) -> pd.DataFrame:
            Reformat a pandas DataFrame by selecting specified columns and
            filling in missing columns with NaN values.

        immoweb_scraping_pipeline(path: str, kind_of_apartment: str, last_page: int,
        columns_to_keep: List[str]) -> pd.DataFrame:
            Run a data scraping and processing pipeline for Immoweb real estate listings.
    """

    def __init__(self, session: HTMLSession):
        self.session = session

    def get_last_page_number_from_url(self, url: str) -> int:
        """
        Retrieve the number of the last page from a given Immoweb search results
        page URL.

        Args:
            url (str): The URL of an Immoweb search results page for houses or
                apartments for rent or sale.
            session (requests_html.HTMLSession): An HTMLSession object for making
                HTTP requests.

        Returns:
            int: The number of the last page of search results.

        Note:
            This function fetches an Immoweb search results page, extracts page
            number elements, and identifies the largest page number to determine
            the last page of search results. It is designed to work with URLs like:

            - https://www.immoweb.be/en/search/house/for-rent?countries=BE&page=1&orderBy=relevance
            - https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance

        Example:
            To retrieve the last page number from an Immoweb search results page:

            >>> url = 'https://www.immoweb.be/en/search/house/for-rent?countries=BE&page=1&orderBy=relevance'
            >>> last_page = get_last_page_number_from_url(url, session)
            >>> print(last_page)

            This function will return the number of the last page, such as 100,
            indicating the total number of pages of results.
        """
        r = self.session.get(url)
        r.html.render(sleep=5)

        elements = r.html.find("span.button__label")

        all_page_numbers = [element.text for element in elements]

        all_numbers = [item for item in all_page_numbers if re.match(r"\d+$", item)]
        largest_number = max(map(int, all_numbers))

        return largest_number

    def get_links_to_listings(self, url: str) -> Element:
        """
        Get the dynamic HTML content of a web page using a headless browser.

        Args:
            url (str): The URL of the web page to fetch and render.
            session (requests_html.HTMLSession): An HTMLSession object for making
                HTTP requests.

        Returns:
            element: The first HTML element found matching the specified XPath
                expression on the rendered page.

        Example:
            To retrieve dynamic HTML content from a web page:

            >>> content_element = get_dynamic_html_content(
            ...     "https://example.com", session
            ... )
            >>> if content_element:
            ...     # Process the retrieved HTML element
            ...     print(content_element.text)
            ... else:
            ...     print("Element not found on the page.")
        """
        r = self.session.get(url)
        r.html.render(sleep=1)
        return r.html.xpath(
            '//*[@id="searchResults"]/div[4]/div/div[1]/div[1]/div[1]', first=True
        )

    def extract_ads_from_given_page(self, links: Set[str]) -> List[pd.DataFrame]:
        """
        Extract and process HTML tables from a list of web page links.

        Args:
            links (set): A set of absolute URLs pointing to web pages containing HTML tables.
            session (requests_html.HTMLSession): An HTMLSession object for making HTTP requests.

        Returns:
            list of DataFrame: A list of DataFrames, each containing a processed HTML table from a web page.

        Example:
            To extract and process tables from a set of web page links:

            >>> links = {
            ...     'https://example.com/page1',
            ...     'https://example.com/page2',
            ...     'https://example.com/page3',
            ... }
            >>> extracted_data = extract_dataframes_from_links(links, session)
            >>> for df in extracted_data:
            ...     # Process each DataFrame as needed
            ...     print(df.head())
        """
        all_ads_from_given_page = []
        for item in list(links.absolute_links):
            try:
                r = self.session.get(item)

                individual_ad = (
                    pd.concat(pd.read_html(StringIO(r.text))).dropna().set_index(0)
                )
                individual_ad.loc["day_of_retrieval", 1] = pd.Timestamp.now()
                individual_ad.loc["ad_url", 1] = item

                all_ads_from_given_page.append(individual_ad)
            except:
                pass
        dfs = [
            df.rename(columns={1: f"source_{i}"})
            for i, df in enumerate(all_ads_from_given_page)
        ]
        return dfs

    @staticmethod
    def join_and_transpose_dataframes(dfs: List[pd.DataFrame]) -> pd.DataFrame:
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
            >>> result_df = join_and_transpose_dataframes(df_list)
            >>> print(result_df)

               0  1
            A  1  2
            B  4  5
            C  7  8
            D 10 11
        """
        return dfs[0].join(dfs[1:]).transpose()

    @staticmethod
    def reformat_scraped_dataframe(
        df: pd.DataFrame, columns_to_keep: List[str]
    ) -> pd.DataFrame:
        """
        Reformat a pandas DataFrame by selecting specified columns and
        filling in missing columns with NaN values.

        Parameters:
        - df (pandas.DataFrame): The original DataFrame to be reformatted.
        - columns_to_keep (list of str): A list of column names to be
        retained in the reformatted DataFrame.

        Returns:
        pandas.DataFrame: A reformatted DataFrame containing the specified columns
        and NaN values in missing columns.

        Example:
        >>> original_df = pd.DataFrame({'column1': [1, 2, 3], 'column2': [4, 5, 6]})
        >>> columns_to_keep = ['column1', 'column2', 'column3']
        >>> reformat_scraped_dataframe(original_df, columns_to_keep)
           column1  column2  column3
        0        1        4      NaN
        1        2        5      NaN
        2        3        6      NaN
        """
        return df.filter(columns_to_keep).pipe(
            lambda df: df.assign(
                **{
                    col: pd.Series(dtype="float64")
                    for col in set(columns_to_keep) - set(df.columns)
                }
            )
        )

    def immoweb_scraping_pipeline(
        self,
        path: str,
        kind_of_apartment: str,
        last_page: int,
        columns_to_keep: List[str],
    ) -> pd.DataFrame:
        """
        Run a data scraping and processing pipeline for Immoweb real estate listings.

        Args:
            path (str): The directory path where CSV files will be saved.
            kind_of_apartment (str): Specifies the type of apartment, either
                'for_rent' or 'for_sale'.
            last_page (int): The number of the last page to scrape.
            session (requests_html.HTMLSession): An HTMLSession object for making
                HTTP requests.

        Returns:
            DataFrame: A DataFrame containing the complete dataset from the
                scraped and processed data.

        Note:
            This function scrapes real estate listings from Immoweb, including
            details about houses or apartments, and saves the data to CSV files in
            the specified directory. It iterates through multiple pages, fetches
            data, joins tables, and creates a complete dataset. The
            `kind_of_apartment` parameter determines whether to fetch listings for
            rent or sale.

        Example:
            To scrape and process real estate listings for rent up to page 106 and
            save CSV files in a specific directory:

            >>> complete_data = immoweb_scraping_pipeline(
            ...     'C:/Users/User/Documents/Data', 'for_rent', 107, session
            ... )
            >>> print(complete_data.head())

            This function will save intermediate and complete datasets to CSV files
            in the specified directory and print 'Task is completed!' when
            finished.
        """
        filepath = Path(path)
        all_tables = []

        for page in tqdm(
            range(1, last_page)
        ):  # Adjust the range if you want to scrape specific pages
            # Generate the URL for the current page
            url = f"https://www.immoweb.be/en/search/house/{kind_of_apartment}?countries=BE&page={page}&orderBy=relevance"

            # Fetch and render the page, then extract links
            links = self.get_links_to_listings(url)

            # Parse data from the retrieved links
            parsed_data = self.extract_ads_from_given_page(links)

            # Join the parsed data into a single table
            joined_tables = self.join_and_transpose_dataframes(parsed_data)

            reformatted_dataframe = self.reformat_scraped_dataframe(
                joined_tables, columns_to_keep
            )

            # Save the joined table to a CSV file
            reformatted_dataframe.to_parquet(
                filepath.joinpath(
                    f"listings_on_page_{page}_{str(pd.Timestamp.now())[:10]}.parquet.gzip"
                ),
                compression="gzip",
                index=False,
            )

            all_tables.append(reformatted_dataframe)

            # Add a sleep duration to avoid overloading the server with requests
            time.sleep(2)

        complete_dataset = pd.concat(all_tables, axis=0)

        complete_dataset.to_parquet(
            filepath.joinpath(
                f"complete_dataset_{str(pd.Timestamp.now())[:10]}.parquet.gzip"
            ),
            compression="gzip",
            index=False,
        )

        print("Task is completed!")
        return complete_dataset


# https://github.com/psf/requests-html/issues/275#issuecomment-513992564
session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)

last_page = 10
path = utils.Configuration.RAW_DATA_PATH
kind_of_apartment = "for_sale"
columns_to_keep = utils.Configuration.features_to_keep_sales


def main():
    # Create an instance of the ImmowebScraper class
    scraper = ImmowebScraper(session)
    # Run the data scraping and processing pipeline
    scraper.immoweb_scraping_pipeline(
        path, kind_of_apartment, last_page, columns_to_keep
    )


if __name__ == "__main__":
    main()
