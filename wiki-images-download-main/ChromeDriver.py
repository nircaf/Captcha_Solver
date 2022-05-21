import time

from selenium import webdriver



driver = webdriver.Chrome('C:\\Users\\ncafr\\AppData\\Local\\Programs\\Chrome_Driver\\chromedriver.exe')  # Optional argument, if not specified will search path.

driver.get('file:///C:/NeuroBrave/Scrapy/RK-Termin%20-%20Kategorie.html');

time.sleep(2) # Let the user actually see something!

# search_box = driver.find_element_by_name('q')
#
# search_box.send_keys('ChromeDriver')

# search_box.submit()
driver.maximize_window()

time.sleep(5) # Let the user actually see something!
button = driver.find_element_by_link_text("Load another picture")
button.click()

driver.quit()