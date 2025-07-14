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

# ✅ Chrome 옵션
options = Options()
options.add_argument("--headless")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--lang=ko-KR")

# ✅ WebDriver 설정
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)
wait = WebDriverWait(driver, 10)

# ✅ 맥도날드 매장 페이지 접속
url = "https://www.mcdonalds.co.kr/kor/store/list.do"
driver.get(url)
time.sleep(3)

all_stores = set()
max_page_visited = 0  # 가장 큰 페이지 번호 기억
stop_flag = False

while not stop_flag:
    print("\n[INFO] 현재 페이징 블록 크롤링 시작")

    # 현재 페이지 소스 파싱
    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")

    # ✅ 현재 활성(선택된) 페이지 수집
    btn_paging = soup.find("div", class_="btnPaging")
    if not btn_paging:
        print("[WARN] 페이지네이션을 찾지 못했습니다. 종료.")
        break

    selected_page = btn_paging.select_one("span.num a[aria-selected='true']")
    current_page_num = None
    if selected_page and selected_page.text.strip().isdigit():
        current_page_num = int(selected_page.text.strip())
        print(f"[INFO] 현재 활성 페이지 감지: {current_page_num}")

        if current_page_num <= max_page_visited:
            print(f"[INFO] 이미 방문한 페이지({current_page_num})가 max({max_page_visited})보다 같거나 작음. 종료.")
            stop_flag = True
            break

        # ✅ 현재 페이지는 이미 열린 상태 → 바로 파싱
        print(f"[INFO] 현재 페이지 {current_page_num} 파싱 중 (버튼 클릭 없이).")
        page_html = driver.page_source
        page_soup = BeautifulSoup(page_html, "html.parser")
        table = page_soup.find("table", class_="tableType01")
        if table:
            rows = table.find_all("tr")
            for row in rows:
                td_name = row.find("td", class_="tdName")
                if not td_name:
                    continue

                dl = td_name.find("dl", class_="name")
                if not dl:
                    continue

                dt = dl.find("dt")
                dd = dl.find("dd")

                store_name = dt.text.strip() if dt else ""
                store_addr = dd.text.strip() if dd else ""

                if store_name and store_addr:
                    all_stores.add((store_name, store_addr))

        max_page_visited = current_page_num

    page_links = btn_paging.select("span.num a")
    current_page_numbers = []
    for a in page_links:
        if "aria-selected" in a.attrs:
            continue
        text = a.text.strip()
        if text.isdigit():
            current_page_numbers.append(int(text))

    print(f"[INFO] 이번 블록에서 탐색할 페이지 번호: {sorted(current_page_numbers)}")

    # ✅ 이번 블록의 각 페이지 번호 클릭 → 파싱
    for page_num in sorted(current_page_numbers):
        if page_num <= max_page_visited:
            print(f"[INFO] 이미 방문한 페이지({page_num})가 max({max_page_visited})보다 같거나 작음. 종료.")
            stop_flag = True
            break

        print(f"[INFO] 페이지 {page_num} 이동 중...")
        driver.execute_script(f"page({page_num});")
        time.sleep(2.5)

        # 새 페이지 파싱
        page_html = driver.page_source
        page_soup = BeautifulSoup(page_html, "html.parser")
        table = page_soup.find("table", class_="tableType01")
        if not table:
            print("[WARN] 테이블이 없습니다. 건너뜀.")
            continue

        rows = table.find_all("tr")
        for row in rows:
            td_name = row.find("td", class_="tdName")
            if not td_name:
                continue

            dl = td_name.find("dl", class_="name")
            if not dl:
                continue

            dt = dl.find("dt").find("strong", class_="tit")
            dd = dl.find("dd")

            store_name = dt.text.strip() if dt else ""
            store_addr = dd.text.strip() if dd else ""

            if store_name and store_addr:
                all_stores.add((store_name, store_addr))

        # ✅ 페이지 방문 기록 갱신
        max_page_visited = page_num

    if stop_flag:
        print("[INFO] 종료 조건 충족. 크롤링 루프 탈출.")
        break

    # ✅ 다음 블록 버튼 존재 여부
    btn_paging_html = driver.page_source
    btn_paging_soup = BeautifulSoup(btn_paging_html, "html.parser")
    btn_paging_div = btn_paging_soup.find("div", class_="btnPaging")
    next_button = btn_paging_div.find("a", class_="arrow next")

    if next_button and "javascript:page" in next_button["href"]:
        next_page_num = next_button["href"].split("page(")[-1].split(")")[0]
        print(f"[INFO] 다음 블록으로 이동 → page({next_page_num})")
        driver.execute_script(f"page({next_page_num});")
        time.sleep(2.5)
    else:
        print("[INFO] 더 이상 다음 블록이 없습니다. 크롤링 종료.")
        break

# ✅ DataFrame 변환
data = [{"store_name": name, "store_addr": addr} for name, addr in all_stores]
df = pd.DataFrame(data)
print(df.head())

df.to_csv("../data/mcdonalds_stores.csv", index=False, encoding='utf-8-sig')
print(f"\n[INFO] 전체 매장 수집 완료: {len(df)}개")
print("[INFO] CSV 파일 저장 완료: mcdonalds_stores.csv")

driver.quit()
