import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
import os

# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]*://.+'

domain = "openai.com"  # <- put your domain to be crawled
full_url = "https://openai.com/"  # <- put your domain to be crawled with https or http


# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

    # Function to get the hyperlinks from a URL
    '''def get_hyperlinks(self,url):

        # Try to open the URL and read the HTML
        try:
            # Open the URL and read the HTML
            with requests.get(url, allow_redirects=True) as response:

                # If the response is not HTML, return an empty list
                if not response.headers['Content-Type'].startswith("text/html"):
                    return []

                # Decode the HTML
                html = response.content.decode('utf-8')
                print(html)
        except Exception as e:
            print(e)
            return []

        # Create the HTML Parser and then Parse the HTML to get hyperlinks
        parser = HyperlinkParser()
        parser.feed(html)

        return parser.hyperlinks'''

    # Function to get the hyperlinks from a URL
    def get_hyperlinks(self , url):

        # Try to open the URL and read the HTML
        try:
            # Open the URL and read the HTML
            with urllib.request.urlopen(url) as response:

                # If the response is not HTML, return an empty list
                if not response.info().get('Content-Type').startswith("text/html"):
                    return []

                # Decode the HTML
                html = response.read().decode('utf-8')
        except Exception as e:
            print(e)
            return []

        # Create the HTML Parser and then Parse the HTML to get hyperlinks
        parser = HyperlinkParser()
        parser.feed(html)

        return parser.hyperlinks
    # Function to get the hyperlinks from a URL that are within the same domain
    def get_domain_hyperlinks(self,local_domain, url):
        clean_links = []
        for link in set(self.get_hyperlinks(url)):
            clean_link = None  # 是一个临时变量，用于存储清理后的超链接

            # If the link is a URL, check if it is within the same domain
            if re.search(HTTP_URL_PATTERN, link):
                # Parse the URL and check if the domain is the same
                url_obj = urlparse(link)
                if url_obj.netloc == local_domain:
                    clean_link = link

            # If the link is not a URL, check if it is a relative link
            else:
                if link.startswith("/"):
                    link = link[1:]
                elif link.startswith("#") or link.startswith("mailto:"):
                    continue
                clean_link = "https://" + local_domain + "/" + link

            if clean_link is not None:
                if clean_link.endswith("/"):
                    clean_link = clean_link[:-1]
                clean_links.append(clean_link)

        # Return the list of hyperlinks that are within the same domain
        return list(set(clean_links))

    def crawl(self,url):
        # Parse the URL and get the domain
        local_domain = urlparse(url).netloc
        # print("11" + local_domain)

        # Create a queue to store the URLs to crawl
        queue = deque([url])

        # Create a set to store the URLs that have already been seen (no duplicates)
        seen = set([url])
        # print("22" + local_domain)

        # Create a directory to store the text files
        if not os.path.exists("text/"):
            os.mkdir("text/")

        if not os.path.exists("text/" + local_domain + "/"):
            os.mkdir("text/" + local_domain + "/")

        # Create a directory to store the csv files
        if not os.path.exists("processed"):
            os.mkdir("processed")

        # While the queue is not empty, continue crawling
        while queue:

            # 程序从队列中取出下一个 URL
            url = queue.pop()
            print(url)  # 控制台输出该 URL，以便查看程序进展。

            # Save text from the url to a <url>.txt file
            with open('text/' + local_domain + '/' + url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:

                # 程序使用 BeautifulSoup 库获取该页面的 HTML 文本
                soup = BeautifulSoup(requests.get(url).text, "html.parser")

                # 并从中提取出纯文本内容。
                text = soup.get_text()

                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                if ("You need to enable JavaScript to run this app." in text):
                    print("Unable to parse page " + url + " due to JavaScript being required")

                # Otherwise, write the text to the file in the text directory
                f.write(text)

            # Get the hyperlinks from the URL and add them to the queue
            # print("33" + local_domain)
            # parser = HyperlinkParser()
            for link in self.get_domain_hyperlinks(local_domain, url):
                if link not in seen:
                    queue.append(link)
                    seen.add(link)
                    # print("end")

parser = HyperlinkParser()

if __name__ == '__main__':
    parser.crawl(full_url)
