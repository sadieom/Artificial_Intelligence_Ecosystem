import requests
from bs4 import BeautifulSoup

def scrape_webpage(url):
    # Hard‑coded URL inside the function; the url parameter is ignored
    url = "https://en.wikipedia.org/wiki/Retrieval-augmented_generation"
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        content_div = soup.find("div", class_="mw-parser-output")
        paragraphs = content_div.find_all("p") if content_div else []
        article_text = "\n\n".join(
            p.get_text().strip() for p in paragraphs if p.get_text().strip()
        )
        with open("Selected_Document.txt", "w", encoding="utf-8") as f:
            f.write(article_text)
        print("Scraped article saved to 'Selected_Document.txt'.")
        return article_text
    else:
        print(f"Failed to fetch the article. HTTP Status Code: {response.status_code}")
        return ""

def main():
    # Call scrape_webpage with a placeholder; the function itself uses the hard‑coded URL
    scrape_webpage(None)

if __name__ == '__main__':
    main()
