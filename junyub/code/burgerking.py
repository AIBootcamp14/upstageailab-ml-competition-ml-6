from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--lang=ko-KR")

service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

url = "https://www.burgerking.co.kr/store/all"
driver.get(url)

wait = WebDriverWait(driver, 10)
time.sleep(3)  

prev_count = 0

while True:
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    store_items = soup.find_all("li", attrs={"data-v-4b769877": True})
    current_count = len(store_items)


    if current_count == prev_count:
        break

    prev_count = current_count

    try:
        more_button = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button.btn02.btn_more02"))
        )
        more_button.click()
        time.sleep(1.5)
    except Exception as e:
        break

final_html = driver.page_source
final_soup = BeautifulSoup(final_html, "html.parser")
all_store_items = final_soup.find_all("li", attrs={"data-v-4b769877": True})

store_data = []

for item in all_store_items:
    store_name_div = item.find("div", class_="tit_store")
    store_name = store_name_div.text.strip() if store_name_div else ""

    store_addr_div = item.find("p", class_="txt_addr")
    store_addr = store_addr_div.text.strip() if store_addr_div else ""

    if store_name and store_addr:
        store_data.append({
            "store_name": store_name,
            "store_addr": store_addr
        })

df = pd.DataFrame(store_data)
print(df)

df.to_csv("../data/burgerking_stores.csv", index=False, encoding='utf-8-sig')
print("[INFO] CSV 파일로 저장 완료: burgerking_stores.csv")

driver.quit()
