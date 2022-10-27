import requests
from bs4 import BeautifulSoup
import itertools
import time
import re
import json
import random

def href(soup):
    # get all href links from one page 
    href=[]
    # for i in soup.find_all("a",class_="question-hyperlink",href=True):
    #     href.append(i['href'])
    # return href

    for question in soup.find_all("div", class_="s-post-summary--content"):
        link = question.find("a",class_="s-link")
        href.append(link['href'])
    return href

def clean_empty_hrefs(hrefs):
   # remove all empty lists
    list_hrefs=[]
    for i in hrefs:
        if i!=[]:
            list_hrefs.append(i)
    # merge all elemenets in one list
    herfs_list=[]
    for i in list_hrefs:
        for j in i:
            herfs_list.append(j)
    return herfs_list

def add_prefix(herfs_list):
    # rearrage those links who do not have 'https://stackoverflow.com' prefix    
    new_href=[]
    prefix='https://stackoverflow.com'
    for h in herfs_list:
        if 'https' not in h:
            m=str(prefix)+str(h)+"answertab=votes#tab-top"
            new_href.append(m)
        else:
            new_href.append(h+"answertab=votes#tab-top")
    return new_href

def single_page_scraper(url):
    req=requests.get(url=url, headers={"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"})
 
    soup=BeautifulSoup(req.text,"html.parser")
    f = open("single_page_scraper.html", "w")
    f.write(str(soup))
    f.close()
    return soup
    
def single_page_question_answer(url):
    page=single_page_scraper(url)#.find_all("div",class_="post-text",itemprop="text")
    question = page.find_all("meta", property="og:title")
    answer=page.find_all("div", class_="s-prose js-post-body")[1]

    code_in_answer = []
    for data in answer:
        found_tags= re.findall(r'<[^>]+>', str(data))
        for tag in found_tags:
            if tag == "<code>":
                start_index = str(data).find("<code>") + 6
                end_index = str(data).find("</code>")
                code_in_answer.append(str(data)[start_index:end_index])
    
    answer_string = ""
    for code in code_in_answer:
        answer_string += code

    return question[0]['content'],answer_string

def get_entries_for_pages(start_page,end_page):
    soups=[]
    for page in range(start_page,end_page+1):
        req=requests.get(url='https://stackoverflow.com/questions/tagged/python?tab=votes&page={}&pagesize=15'.format(page), headers={"User-Agent": "Mozilla/5.0 (X11; CrOS x86_64 12871.102.0) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.141 Safari/537.36"})
        soup=BeautifulSoup(req.text,"html.parser")
        soups.append(soup)
        time.sleep(2)
        # print("Pages pulled: " + str(len(soups)))  
    
    # obtain all href
    hrefs=[]
    for soup in soups:
        hrefs.append(href(soup))

    herfs_list=clean_empty_hrefs(hrefs)
    new_hrefs_list=add_prefix(herfs_list)

    # print("Obtaining Questions")
    quesitons=[]
    answers=[]
    total_urls = 0
    excepted_urls = 0
    for url in new_hrefs_list:
        total_urls += 1
        random_sleep_time = random.randint(5,50)
        if random_sleep_time > 40:
            random_sleep_time += random.randint(0,50)
        time.sleep(random_sleep_time/10)
        print("Processing url " + str(total_urls) + " waited for " + str(random_sleep_time/10) + " seconds")
        try:
            q,a=single_page_question_answer(url)
            if a != "":
                quesitons.append(q)
                answers.append(a)
                print("Added Entry: " + str(q))
        except:
            excepted_urls += 1    
    
    # print("quesitons and answers are ready!")
    # print(str(excepted_urls) + " " + str(total_urls))

    entries = []
    for i in range(len(quesitons)):
        entry = {"idx": "webquery-test-" + str(i+1), "doc": str(quesitons[i]), "code": str(answers[i])} 
        entries.append(entry)

    return entries
   
def get_data(target_size, nr_pages_per_iteration):
    page_counter = 1
    total_entries = 0
    entries = []
    while total_entries < target_size:
        print("Processing page " + str(page_counter) + " to " + str(page_counter + nr_pages_per_iteration - 1))
        entries += get_entries_for_pages(page_counter, page_counter + nr_pages_per_iteration - 1)
        print("Total entries: " + str(len(entries)))

        f = open("serious_run_python_" + str(len(entries)) + ".json", "w")
        json_entries = json.dumps(entries)
        f.write(str(json_entries))
        f.close()
       
        page_counter += nr_pages_per_iteration
        if page_counter > 10000:
            break
    

get_data(1000,2)
