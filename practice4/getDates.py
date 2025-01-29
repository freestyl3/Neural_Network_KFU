from datetime import date, timedelta
import json

def get_first_and_last_days(start_year, end_year):
    months = ['January', 'February', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']
    dates = []

    first_day = date(start_year, 1, 1)
    end_day = date(end_year, 12, 31)

    while first_day <= end_day:
        next_month = first_day.replace(day=28) + timedelta(days=4)
        last_day = next_month.replace(day=1) - timedelta(days=1)

        dates.append(
            {
                'start_date': date.strftime(first_day, '%Y-%m-%d'),
                'end_date': date.strftime(last_day, '%Y-%m-%d'),
                'month': months[(first_day.month - 1) % 12],
                'year': first_day.year
            }
        )

        first_day = last_day + timedelta(days=1)

    return dates


# Пример использования
if __name__ == '__main__':
    dates = get_first_and_last_days(2023, 2024)

    print(json.dumps(dates, indent=4))

    for date in dates:
        print(f'{date['month']}{date['year']}.json')

