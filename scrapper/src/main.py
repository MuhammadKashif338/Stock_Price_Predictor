import time
import csv
import os

from undetected_chromedriver import Chrome

from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

root = os.path.dirname(__file__)[:-4]

chrome_options = Options()
chrome_options.binary_location = "/opt/google/chrome/google-chrome"
chrome_options.add_argument("--headless")

user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"

chrome_options.add_argument(f"--user-agent={user_agent}")

DRIVER = Chrome(options=chrome_options, version_main=125)
PAGE_LOAD_DELAY = 3
DELAY = 1.5
WAIT = WebDriverWait(DRIVER, DELAY)

URL = "https://dps.psx.com.pk/historical"

DRIVER.get(URL)
time.sleep(5)

inputBoxSymbol = WAIT.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "#historicalSymbolSearch"))
)

selectMonth = WAIT.until(
    EC.presence_of_element_located(
        (By.CSS_SELECTOR, "div.dropdown.historical__month select")
    )
)
selectMonth = Select(selectMonth)

selectYear = WAIT.until(
    EC.presence_of_element_located(
        (By.CSS_SELECTOR, "div.dropdown.historical__year select")
    )
)
selectYear = Select(selectYear)

btnSubmit = WAIT.until(
    EC.presence_of_element_located((By.CSS_SELECTOR, "#historicalSymbolBtn"))
)


def write_csv(data, file_path, fieldnames):
    dest = root + "/" + os.path.dirname(file_path)
    if not os.path.exists(dest):
        os.mkdir(dest)

    with open(file_path, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(fieldnames)
        writer.writerows(data)


def parse_table(year, month):
    table = WAIT.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#historicalTable"))
    )
    table_head = table.find_element(By.CSS_SELECTOR, "thead.tbl__head")

    header = [
        column.text.strip() for column in table_head.find_elements(By.TAG_NAME, "th")
    ]
    header = ["YEAR", "MONTH", "DAY"] + header[1:]
    print(header)

    master = []
    rows = table.find_elements(By.CSS_SELECTOR, "tbody.tbl__body tr")
    for row in rows:
        data = [
            column.text.strip() for column in row.find_elements(By.CSS_SELECTOR, "td")
        ]
        data[0:1] = [year, month, int(data[0].split()[1][:-1])]
        for i in range(3, 7):
            data[i] = float(data[i])
        data[-1] = int(data[-1].replace(",", ""))
        print(data)
        master.append(data)

    info = WAIT.until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "#historicalTable_info"))
    )
    tokens = info.text.strip().split()
    print(tokens[3], "out of", tokens[5])
    assert tokens[3] == tokens[5], "Missing rows"

    write_csv(master, f"out/{year}/{month}.csv", fieldnames=header)


def main():
    inputBoxSymbol.send_keys("SYS")

    # For training data
    # years = [2020, 2021, 2022, 2023]
    # months = range(1, 13)

    # For testing data
    years = [2024]
    months = range(1, 7)

    for year in years:
        selectYear.select_by_value(str(year))
        for month in months:
            selectMonth.select_by_value(str(month))
            DRIVER.execute_script("arguments[0].click();", btnSubmit)
            time.sleep(3)
            parse_table(year, month)
            print(year, month)
            time.sleep(1)


if __name__ == "__main__":
    main()
