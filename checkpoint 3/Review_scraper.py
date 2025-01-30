import requests
from bs4 import BeautifulSoup
import csv
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from time import sleep

def fetch_page(url):
    response = requests.get(url)
    return response.text if response.status_code == 200 else None

def parse_static_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    reviews = []
    review_elements = soup.find_all('div', class_='review')
    
    for review in review_elements:
        review_text = review.find('p', class_='review-text')
        reviewer_name = review.find('span', class_='reviewer-name')
        review_date = review.find('span', class_='review-date')
        rating = review.find('span', class_='rating')

        reviews.append({
            'Review': review_text.get_text(strip=True) if review_text else '',
            'Reviewer': reviewer_name.get_text(strip=True) if reviewer_name else '',
            'Date': review_date.get_text(strip=True) if review_date else '',
            'Rating': rating.get_text(strip=True) if rating else ''
        })
    
    return reviews

def parse_dynamic_page(url):
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
    driver.get(url)
    sleep(5)
    
    reviews = []
    review_elements = driver.find_elements(By.CLASS_NAME, 'review')
    
    for review in review_elements:
        try:
            review_text = review.find_element(By.CLASS_NAME, 'review-text').text
            reviewer_name = review.find_element(By.CLASS_NAME, 'reviewer-name').text
            review_date = review.find_element(By.CLASS_NAME, 'review-date').text
            rating = review.find_element(By.CLASS_NAME, 'rating').text

            reviews.append({
                'Review': review_text,
                'Reviewer': reviewer_name,
                'Date': review_date,
                'Rating': rating
            })
        except:
            pass
    
    driver.quit()
    return reviews

def save_reviews_to_csv(reviews, filename='Reviews.csv'):
    if reviews:
        keys = reviews[0].keys()
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=keys)
            writer.writeheader()
            writer.writerows(reviews)

def main():
    url = input("Enter the URL of the product reviews page: ")
    html = fetch_page(url)
    
    if html:
        reviews = parse_static_page(html) or parse_dynamic_page(url)
        save_reviews_to_csv(reviews)

if __name__ == '__main__':
    main()
