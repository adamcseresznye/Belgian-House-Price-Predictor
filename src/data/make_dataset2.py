import logging
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
    def __init__(
        self,
        session: HTMLSession,
        last_page: int,
        start_page: int = 1,
        kind_of_apartment: str = "for_sale",
    ):
        self.session = session
        self.start_page = start_page
        self.last_page = last_page
        self.path = utils.Configuration.RAW_DATA_PATH
        self.kind_of_apartment = kind_of_apartment
        self.features_to_keep = utils.Configuration.features_to_keep_sales

        # Construct the absolute path for the log file using Path
        self.log_file_path = self.path / "error.log"

        # Set up logging
        logging.basicConfig(
            filename=self.log_file_path,
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def get_last_page_number_from_url(self, url: str) -> int:
        r = self.session.get(url)
        r.html.render(sleep=5)

        elements = r.html.find("span.button__label")

        all_page_numbers = [element.text for element in elements]

        all_numbers = [item for item in all_page_numbers if re.match(r"\d+$", item)]
        largest_number = max(map(int, all_numbers))

        return largest_number

    def get_links_to_listings(self, url: str) -> Element:
        r = self.session.get(url)
        r.html.render(sleep=1)
        return r.html.xpath(
            '//*[@id="searchResults"]/div[4]/div/div[1]/div[1]/div[1]', first=True
        )

    def extract_ads_from_given_page(
        self,
        links: Set[str],
        filepath: str,
        save_to_disk: bool = False,
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
            except Exception as e:
                if "No tables found" in str(e):
                    logging.error(f"No tables found while processing {item}")
                    pass  # Continue processing the next ad
                elif "cannot reindex on an axis with duplicate labels" in str(e):
                    logging.error(f"Duplicate labels found while processing {item}")
                    pass  # Continue processing the next ad
                else:
                    raise e  # Raise the error if it's not one of the expected errors

        all_ads_from_given_page_df = pd.concat(all_ads_from_given_page, axis=0)

        if save_to_disk:
            all_ads_from_given_page_df.to_parquet(
                filepath.joinpath(
                    f"listings_on_page_{number}_{str(pd.Timestamp.now())[:10]}.parquet.gzip"
                ),
                compression="gzip",
                index=False,
            )

        return all_ads_from_given_page_df

    def immoweb_scraping_pipeline(self, kind_of_apartment: str) -> pd.DataFrame:
        filepath = Path(self.path)
        all_tables = []
        print(f"start_page: {self.start_page}, last_page: {self.last_page}")
        for page in tqdm(range(self.start_page, self.last_page)):
            url = f"https://www.immoweb.be/en/search/house/{kind_of_apartment}?countries=BE&page={page}&orderBy=relevance"

            # Fetch and render the page, then extract links
            links = self.get_links_to_listings(url)

            # Parse data from the retrieved links
            parsed_data = self.extract_ads_from_given_page(links, self.path)

            all_tables.append(parsed_data)

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

path = utils.Configuration.RAW_DATA_PATH
kind_of_apartment = "for_sale"


def main():
    # Create an instance of the ImmowebScraper class
    scraper = ImmowebScraper(session, last_page=3)
    # Run the data scraping and processing pipeline
    scraper.immoweb_scraping_pipeline(
        kind_of_apartment,
    )


if __name__ == "__main__":
    main()
