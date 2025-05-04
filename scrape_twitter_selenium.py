from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
import pandas as pd

# Setup driver
path_to_driver = r"D:\chromedriver-win64\chromedriver.exe"
service = Service(executable_path=path_to_driver)
options = webdriver.ChromeOptions()
options.add_argument("--start-maximized")
driver = webdriver.Chrome(service=service, options=options)

# Manual login
print("Silakan login ke Twitter di browser...")
driver.get("https://twitter.com/login")
input("Setelah login selesai, tekan ENTER di terminal ini...")

# Masuk ke pencarian
query = "guru lalai"
search_url = f"https://twitter.com/search?q={query}&src=typed_query&f=live"
driver.get(search_url)
time.sleep(10)

# Inisialisasi
data = []
seen = set()
max_scrolls = 2000
scroll_pause = 5
last_height = driver.execute_script("return document.body.scrollHeight")

for scroll in range(max_scrolls):
    print(f"📜 Scroll ke-{scroll + 1}")
    time.sleep(scroll_pause)

    tweet_blocks = driver.find_elements(By.XPATH, '//article[@data-testid="tweet"]')

    for tweet in tweet_blocks:
        try:
            WebDriverWait(tweet, 2).until(
                EC.presence_of_element_located((By.XPATH, './/div[@lang]'))
            )
            text_elem = tweet.find_element(By.XPATH, './/div[@lang]')
            text = text_elem.text

            time_elem = tweet.find_element(By.XPATH, './/time')
            timestamp = time_elem.get_attribute("datetime")

            key = f"{text}_{timestamp}"
            if text.strip() and key not in seen:
                seen.add(key)
                data.append({
                    "platform": "twitter",
                    "komentar": text,
                    "timestamp": timestamp,
                    "sentimen": ""
                })

        except Exception:
            continue

    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(4)
    new_height = driver.execute_script("return document.body.scrollHeight")
    if new_height == last_height:
        print("⛔ Tidak ada perubahan tinggi halaman, berhenti scroll.")
        break
    last_height = new_height

# Simpan hasil
df = pd.DataFrame(data)
df.to_csv("komentar_twitter3.csv", index=False)
print(f"✅ Berhasil simpan {len(df)} komentar unik ke komentar_twitter.csv")

driver.quit()
