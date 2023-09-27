import sys

import pandas as pd
from requests_html import HTMLSession

from data import make_dataset, utils

session = HTMLSession(
    browser_args=[
        "--no-sandbox",
        "--user-agent=Mozilla/5.0 (Windows NT 5.1; rv:7.0.1) Gecko/20100101 Firefox/7.0.1",
    ]
)


def main():
    # Use the get_last_page_number_from_url function to retrieve the last page number
    last_page_number = make_dataset.get_last_page_number_from_url()
    # Create an instance of the ImmowebScraper class
    scraper = make_dataset.ImmowebScraper(session, last_page=last_page_number)
    # Run the data scraping and processing pipeline
    scraped_dataset = scraper.immoweb_scraping_pipeline()

    df_pre_processed = make_dataset.pre_process_dataframe(scraped_dataset)
    df_address_separated = make_dataset.separate_address(df_pre_processed)
    make_dataset.filter_out_missing_indexes(df_address_separated)


if __name__ == "__main__":
    main()
