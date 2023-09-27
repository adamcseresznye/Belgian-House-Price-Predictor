import logging
import re
import sys
import time
from enum import Enum
from io import StringIO
from typing import List, Set

import pandas as pd
from requests_html import Element, HTMLSession
from tqdm import tqdm

from data import utils


class LastPage(Enum):
    LASTPAGE: str = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance"


def get_last_page_number_from_url(
    url: str = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance",
) -> int:
    """
    Retrieve the last page number from an Immoweb search URL.

    This function sends a GET request to the specified URL, extracts the page numbers from
    the rendered HTML, and returns the largest page number found.

    Args:
        url (str, optional): The URL to query. Defaults to the Immoweb search URL.

    Returns:
        int: The largest page number found on the page.

    Example:
        To get the last page number of a search for houses for sale in Belgium:
        >>> url = "https://www.immoweb.be/en/search/house/for-sale?countries=BE&page=1&orderBy=relevance"
        >>> last_page = get_last_page_number_from_url(url)
        >>> print(last_page)
        10
    """
    r = session.get(url)
    r.html.render(sleep=5)

    elements = r.html.find("span.button__label")

    all_page_numbers = [element.text for element in elements]

    all_numbers = [item for item in all_page_numbers if re.match(r"\d+$", item)]
    largest_number = max(map(int, all_numbers))

    return largest_number


class ImmowebScraper:
    """
    A class for scraping and processing data from Immoweb.

    Attributes:
        session (HTMLSession): The HTMLSession used for web scraping.
        start_page (int): The starting page number.
        last_page (int): The last page number to scrape.
        kind_of_apartment (str): The type of apartment to search for.
        save_to_disk (bool): Whether to save data to disk.
    """

    def __init__(
        self,
        session: HTMLSession,
        last_page: int,
        start_page: int = 1,
        kind_of_apartment: str = "for_sale",
        save_to_disk: bool = False,
    ):
        """
        Initialize the ImmowebScraper.

        Args:
            session (HTMLSession): The HTMLSession used for web scraping.
            last_page (int): The last page number to scrape.
            start_page (int, optional): The starting page number. Defaults to 1.
            kind_of_apartment (str, optional): The type of apartment to search for.
                Defaults to "for_sale".
            save_to_disk (bool, optional): Whether to save data to disk. Defaults to False.
        """
        self.session = session
        self.start_page = start_page
        self.last_page = last_page
        self.kind_of_apartment = kind_of_apartment
        self.features_to_keep = utils.Configuration.features_to_keep_sales
        self.path = utils.Configuration.RAW_DATA_PATH
        self.save_to_disk = save_to_disk

        # Construct the absolute path for the log file using Path
        self.log_file_path = self.path / "make_dataset_error.log"

        # Set up logging
        self.setup_logging()

    def setup_logging(self):
        """Set up logging configuration."""
        logging.basicConfig(
            filename=self.log_file_path,
            filemode="w",
            level=logging.WARNING,
            format="%(asctime)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",  # Add timestamp format
        )

    def get_links_to_listings(self, url: str) -> Element:
        r = self.session.get(url)
        r.html.render(sleep=1)
        return r.html.xpath(
            '//*[@id="searchResults"]/div[4]/div/div[1]/div[1]/div[1]', first=True
        )

    def extract_ads_from_given_page(
        self,
        links: Set[str],
    ) -> List[pd.DataFrame]:
        all_ads_from_given_page = []
        for number, item in enumerate(list(links.absolute_links)):
            try:
                r = self.session.get(item)

                individual_ad = (
                    pd.concat(pd.read_html(StringIO(r.text))).dropna().set_index(0)
                )
                individual_ad.loc["day_of_retrieval", 1] = pd.Timestamp.now()
                individual_ad.loc["ad_url", 1] = item

                individual_ad_revised = (
                    individual_ad.transpose()
                    .filter(self.features_to_keep)
                    .pipe(
                        lambda df: df.assign(
                            **{
                                col: pd.Series(dtype="float64")
                                for col in set(self.features_to_keep) - set(df.columns)
                            }
                        )
                    )
                )

                all_ads_from_given_page.append(individual_ad_revised)

                # Save data to disk for each page if save_to_disk is True
                if self.save_to_disk:
                    self.save_data_to_disk(number, individual_ad_revised)
            except Exception as e:
                self.handle_extraction_error(e, item)
                continue

        all_ads_from_given_page_df = pd.concat(all_ads_from_given_page, axis=0)

        # Always save the complete dataset to disk
        self.save_complete_dataset(all_ads_from_given_page_df)

        return all_ads_from_given_page_df

    def handle_extraction_error(self, error: Exception, item: str):
        """Handle exceptions during data extraction."""
        if "No tables found" in str(error):
            logging.error(f"No tables found while processing {item}")
        elif "cannot reindex on an axis with duplicate labels" in str(error):
            logging.error(f"Duplicate labels found while processing {item}")
        elif "Expected bytes" in str(error):
            logging.error(f"Conversion in [0] failed while processing {item}")
        elif "None of [0] are in the columns" in str(error):
            logging.error(f"Empty data while processing {item}")
        elif "Could not convert" in str(error):
            logging.error(f"Conversion in [1] failed while processing {item}")
        else:
            raise error

    def save_data_to_disk(self, number: int, data: pd.DataFrame):
        """Save data to disk."""
        filepath = (
            self.path
            / f"listings_on_page_{number}_{str(pd.Timestamp.now())[:10]}.parquet.gzip"
        )
        data.to_parquet(filepath, compression="gzip", index=False)

    def save_complete_dataset(self, data: pd.DataFrame):
        """Save the complete dataset to disk."""
        filepath = (
            self.path / f"complete_dataset_{str(pd.Timestamp.now())[:10]}.parquet.gzip"
        )
        data.to_parquet(filepath, compression="gzip", index=False)

    def __repr__(self):
        return f"ImmowebScraper(start_page={self.start_page}, end_page={self.last_page}, kind_of_apartment={self.kind_of_apartment}, save_to_disk={self.save_to_disk})"

    def immoweb_scraping_pipeline(self) -> pd.DataFrame:
        """
        Execute the Immoweb data scraping and processing pipeline.

        This method performs a series of steps to scrape and process data from Immoweb,
        including fetching listings, parsing information, and saving the dataset.

        Returns:
            pd.DataFrame: The complete dataset.

        Raises:
            Exception: If any error occurs during the pipeline execution.

        Example:
            To run the Immoweb scraping pipeline and obtain the dataset:
            >>> scraper = ImmowebScraper()
            >>> dataset = scraper.immoweb_scraping_pipeline()
            >>> print(dataset.head())

        Notes:
            - The dataset is saved to the path specified in 'utils.Configuration.RAW_DATA_PATH'.
            - A delay is introduced between page requests to avoid overloading the server.
            - Errors during the pipeline are logged, and the pipeline continues with the next page.
        """
        all_tables = []
        complete_dataset = pd.DataFrame()
        print(f"start_page: {self.start_page}, last_page: {self.last_page - 1}")

        try:
            for page in tqdm(range(self.start_page, self.last_page)):
                url = f"https://www.immoweb.be/en/search/house/{self.kind_of_apartment}?countries=BE&page={page}&orderBy=relevance"

                try:
                    # Fetch and render the page, then extract links
                    links = self.get_links_to_listings(url)

                    # Parse data from the retrieved links
                    parsed_data = self.extract_ads_from_given_page(links)

                    all_tables.append(parsed_data)

                    # Add a sleep duration to avoid overloading the server with requests
                    time.sleep(2)
                except Exception as page_error:
                    # Log the error for the specific page
                    logging.error(
                        f"An error occurred on page {page}: {str(page_error)}"
                    )
                    continue  # Continue with the next page

            complete_dataset = pd.concat(all_tables, axis=0)

            # Save complete dataset to disk
            self.save_complete_dataset(complete_dataset)

            print("Task is completed!")
        except Exception as pipeline_error:
            # Log the error for the entire pipeline
            logging.error(f"An error occurred in the pipeline: {str(pipeline_error)}")
            print("Webscraping is terminating.")
            # sys.exit(1)  # Exit with a non-zero exit code to indicate an error
            pass
        return complete_dataset


# https://github.com/psf/requests-html/issues/275#issuecomment-513992564
session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)


def main():
    try:
        # Use the get_last_page_number_from_url function to retrieve the last page number
        last_page_number = get_last_page_number_from_url()
        # Create an instance of the ImmowebScraper class
        scraper = ImmowebScraper(session, last_page=last_page_number)
        # Run the data scraping and processing pipeline
        scraper.immoweb_scraping_pipeline()
    except Exception as e:
        # Handle exceptions here, and exit with a non-zero exit code
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
