import pandas as pd
import pytest
from requests_html import HTMLSession

from data import make_dataset


# Mock data or setup a session for testing purposes
@pytest.fixture
def test_session():
    return HTMLSession()


@pytest.fixture
def test_scraper(test_session):
    return make_dataset.ImmowebScraper(test_session, last_page=3)


# Test get_last_page_number_from_url method
def test_get_last_page_number_from_url(test_scraper):
    # Mock a response for a test URL
    class MockResponse:
        def __init__(self, text):
            self.text = text

        def render(self, sleep=None):
            pass

    test_scraper.session.get = lambda url: MockResponse("Page 1\nPage 2\nPage 3")
    last_page_number = test_scraper.get_last_page_number_from_url("https://example.com")
    assert isinstance(last_page_number, int)
    assert last_page_number == 3


# Test get_links_to_listings method (requires real HTML)
def test_get_links_to_listings(test_scraper):
    # Mock a response with HTML content for testing
    class MockResponse:
        def __init__(self):
            self.html = HTMLSession().get("https://example.com")
            self.html.render()

    test_scraper.session.get = MockResponse
    links = test_scraper.get_links_to_listings("https://example.com")
    assert isinstance(links, pd.DataFrame)  # Check that it returns a DataFrame


# Test extract_ads_from_given_page method (requires real HTML)
def test_extract_ads_from_given_page(test_scraper):
    # Mock data for test links
    test_links = {
        "https://example.com/1",
        "https://example.com/2",
        "https://example.com/3",
    }

    # Mock a response with HTML content for testing
    class MockResponse:
        def __init__(self):
            self.html = HTMLSession().get("https://example.com")
            self.html.render()

    test_scraper.session.get = MockResponse

    # Mock the save_data_to_disk method for testing
    def mock_save_data_to_disk(number, data):
        pass

    test_scraper.save_data_to_disk = mock_save_data_to_disk

    result = test_scraper.extract_ads_from_given_page(test_links)
    assert isinstance(result, pd.DataFrame)  # Check that it returns a DataFrame


# Test handle_extraction_error method
def test_handle_extraction_error(test_scraper):
    error_message = "No tables found while processing https://example.com"
    with pytest.raises(Exception, match=error_message):
        test_scraper.handle_extraction_error(
            Exception(error_message), "https://example.com"
        )


# Test save_data_to_disk method (requires real data)
def test_save_data_to_disk(test_scraper):
    # Mock data for testing
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test_number = 1

    # Perform the save operation (modify path for testing)
    test_scraper.path = "/path/to/test/data"
    test_scraper.save_data_to_disk(test_number, test_data)

    # Check if the file exists
    filepath = f"/path/to/test/data/listings_on_page_{test_number}*.parquet.gzip"
    assert any(filepath.glob("*.parquet.gzip"))


# Test save_complete_dataset method (requires real data)
def test_save_complete_dataset(test_scraper):
    # Mock data for testing
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    # Perform the save operation (modify path for testing)
    test_scraper.path = "/path/to/test/data"
    test_scraper.save_complete_dataset(test_data)

    # Check if the file exists
    filepath = f"/path/to/test/data/complete_dataset_*.parquet.gzip"
    assert any(filepath.glob("*.parquet.gzip"))


# Test immoweb_scraping_pipeline method (requires real data)
def test_immoweb_scraping_pipeline(test_scraper):
    # Mock the get_links_to_listings method for testing
    def mock_get_links_to_listings(url):
        class MockResponse:
            def __init__(self):
                self.html = HTMLSession().get("https://example.com")
                self.html.render()

        return MockResponse()

    test_scraper.get_links_to_listings = mock_get_links_to_listings

    # Mock the extract_ads_from_given_page method for testing
    def mock_extract_ads_from_given_page(links):
        return pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})

    test_scraper.extract_ads_from_given_page = mock_extract_ads_from_given_page

    # Mock the save_complete_dataset method for testing
    def mock_save_complete_dataset(data):
        pass

    test_scraper.save_complete_dataset = mock_save_complete_dataset

    result = test_scraper.immoweb_scraping_pipeline()
    assert isinstance(result, pd.DataFrame)  # Check that it returns a DataFrame


if __name__ == "__main__":
    pytest.main()
