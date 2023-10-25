from pathlib import Path

import numpy as np
import pandas as pd
import pygsheets

from data import pre_process, utils


def connect_to_google_sheet(
    spreadsheet_name="HousePricePrediction", worksheet_name="Sheet1"
):
    """
    Connect to a Google Sheet using service account authentication.

    Parameters:
        spreadsheet_name (str): The name of the Google Spreadsheet.
        worksheet_name (str): The name of the worksheet in the Spreadsheet.

    Returns:
        pygsheets.worksheet.Worksheet: The connected Google Sheet worksheet.

    This function connects to a Google Sheet using a service account JSON file for authentication.
    It opens the specified spreadsheet and worksheet and returns the connected worksheet.

    Example:
        worksheet = connect_to_google_sheet(spreadsheet_name="MyData", worksheet_name="DataSheet")
    """
    try:
        # Define the path to the service account JSON file
        service_account_json = (
            Path.cwd()
            .joinpath(utils.Configuration.RAW_DATA_PATH)
            .parent.joinpath("housepriceprediction-403017-cdf0b4d6f273.json")
        )

        # Authorize and connect to the Google Sheet
        gc = pygsheets.authorize(service_file=service_account_json)
        spreadsheet = gc.open(spreadsheet_name)
        worksheet = spreadsheet.worksheet_by_title(worksheet_name)

        print("Connected to Google Sheet successfully")
        return worksheet
    except Exception as e:
        print(f"Authentication was not successful. Error: {e}")
        return None


import numpy as np
import pandas as pd
import pygsheets


def update_database_in_chunks(worksheet, df_to_append, chunk_size=1000):
    """
    Update a Google Sheet with data from a DataFrame in chunks.

    Parameters:
        worksheet (pygsheets.worksheet.Worksheet): The target Google Sheet worksheet.
        df_to_append (pd.DataFrame): The DataFrame to append to the Google Sheet.
        chunk_size (int, optional): The size of each data chunk. Default is 1000.

    This function updates a Google Sheet by appending a DataFrame to it in chunks.
    It calculates the number of chunks based on the chunk size, slices the DataFrame
    into chunks, retrieves the existing data, concatenates it with the new data, and
    updates the Google Sheet with the combined data.

    The new data is appended below the existing data to avoid overwriting or duplication.

    Parameters:
        worksheet (pygsheets.worksheet.Worksheet): The target Google Sheet worksheet.
        df_to_append (pd.DataFrame): The DataFrame to append to the Google Sheet.
        chunk_size (int, optional): The size of each data chunk. Default is 1000.

    Example:
        worksheet = connect_to_google_sheet(spreadsheet_name="MyData", worksheet_name="DataSheet")
        data_to append = pd.DataFrame({'Column1': [1, 2, 3], 'Column2': ['A', 'B', 'C']})
        update_database_in_chunks(worksheet, data_to_append, chunk_size=500)
    """
    try:
        # Calculate the number of chunks
        n = len(df_to_append) / chunk_size

        # Slice the DataFrame into chunks and update the Google Sheet
        for i in range(int(np.ceil(n))):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk = df_to_append.iloc[start:end, :]

            # Retrieve the current data from the Google Sheet
            df_current = worksheet.get_as_df()
            position_to_append_to = df_current.shape[0] + 1

            # Update the Google Sheet with the combined data
            worksheet.set_dataframe(chunk, start=(1, 1), extend=True)
            worksheet.sync()

            print(f"Chunk {i + 1} shape: {chunk.shape}")

        print("Database updated successfully")
    except Exception as e:
        print(f"Database update failed. Error: {e}")
