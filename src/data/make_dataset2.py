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
    def __init__(self, session: HTMLSession):
        self.session = session

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

    def extract_ads_from_given_page(self, links: Set[str]) -> List[pd.DataFrame]:
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
        return dfs[0].join(dfs[1:]).transpose()

    @staticmethod
    def reformat_scraped_dataframe(
        df: pd.DataFrame, columns_to_keep: List[str]
    ) -> pd.DataFrame:
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
            reformatted_dataframe.to_csv(
                filepath.joinpath(
                    f"listings_on_page_{page}_{str(pd.Timestamp.now())[:10]}.csv"
                ),
                index=False,
            )

            all_tables.append(reformatted_dataframe)

            # Add a sleep duration to avoid overloading the server with requests
            time.sleep(2)

        complete_dataset = pd.concat(all_tables, axis=0)

        complete_dataset.to_csv(
            filepath.joinpath(f"complete_dataset_{str(pd.Timestamp.now())[:10]}.csv"),
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
last_page = 4
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
