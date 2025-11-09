import requests
from bs4 import BeautifulSoup
import time
import json
import os
from datetime import datetime

# URL of the webpage you want to scrape
url = 'https://konkanrailway.com/VisualTrain/otrktp0100Table.jsp'

# Function to scrape data
def scrape_data():
    # Send a request to the website
    response = requests.get(url)
    response.raise_for_status()

    # Parse the HTML content
    soup = BeautifulSoup(response.text, 'html.parser')

    # Locate the <tr> element containing the timestamp
    timestamp_row = soup.find('tr', {'align': 'right'})
    if timestamp_row:
        timestamp_text = timestamp_row.find('font').text.strip()
        # Extract and format the timestamp
        timestamp = timestamp_text.replace('Last Updated time : ', '').strip()
        # Convert to standard datetime format
        timestamp = datetime.strptime(timestamp, '%d/%m/%Y %H:%M:%S').strftime('%Y-%m-%d %H:%M:%S')
    else:
        # Handle missing timestamp gracefully
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"Warning: Timestamp not found on the page. Using current time: {timestamp}")

    # Find the table containing the train data
    table = soup.find('table', {'id': 'empNoHelpList'})
    if table is None:
        print("Error: Could not find the table with train data.")
        return

    rows = table.find_all('tr')[3:]  # Skip the header rows

    # Prepare data
    data = {}
    data[timestamp] = []

    for row in rows:
        cols = row.find_all('td')
        if len(cols) == 6:
            data[timestamp].append({
                'Train No': cols[0].text.strip(),
                'Train Name': cols[1].text.strip(),
                'Status': cols[2].text.strip(),
                'Station': cols[3].text.strip(),
                'Time': cols[4].text.strip(),
                'Delay': cols[5].text.strip()
            })

    # Save data to a JSON file
    save_to_json(data)

# Function to save data to a JSON file
def save_to_json(data):
    filename = 'train_status.json'
    
    # Check if the file exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'r') as file:
            existing_data = json.load(file)
    else:
        existing_data = {}

    # Update existing data with the new data
    existing_data.update(data)

    # Save updated data back to the file
    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

# Main loop to scrape data every 10 minutes
while True:
    scrape_data()
    time.sleep(180)  # Wait for 10 minutes (600 seconds)
