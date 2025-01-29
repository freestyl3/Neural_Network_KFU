import json
import os
import time

import requests
from dotenv import load_dotenv

from getDates import get_first_and_last_days

load_dotenv()

cities = ['Moscow']

API_KEY = os.getenv('API_KEY')
url = f'https://api.weatherbit.io/v2.0/history/hourly'

def get_historical_weather_data(city, start_date, end_date):
    params = {
        'city': city,
        'start_date': start_date,
        'end_date': end_date,
        'key': API_KEY,
        'unit': 'M',  # В Цельсиях
    }
    response = requests.get(url, params=params)
    print(response)
    data = response.json()
    return data

def collect_data_by_years(start_year, end_year):
    dates = get_first_and_last_days(start_year, end_year)
    for city in cities:
        for date in dates:
            month = date['month']
            year = date['year']
            start_date = date['start_date']
            end_date = date['end_date']
            filename = f'./Data/{city}/{month}{year}.json'
            if not os.path.isfile(filename):
                data = get_historical_weather_data(city, start_date, end_date)
                with open(filename, 'w') as file:
                    json.dump(data, file)
                print(f'{filename} saved!')
                time.sleep(2)
            else:
                print(f'File {month}{year}.json is already exists!')
