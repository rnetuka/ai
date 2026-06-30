import json
import requests
from bs4 import BeautifulSoup
from typing import List
from ddgs import DDGS
from langchain_core.prompts import PromptTemplate
from llm_client import gemini, mistral


def to_obj(string: str) -> object | dict:
    try:
        if string.startswith('```json') and string.endswith('```'):
            string = string.replace('```json', '').replace('```', '')
        return json.loads(string)
    except Exception:
        return {}


def get_prompt_template(name: str) -> PromptTemplate:
    with open(f'resources/prompts/{name}.txt') as file:
        text = file.read()
        return PromptTemplate.from_template(text)


def web_search(query: str, max_results: int) -> List[str]:
    results = DDGS().text(query, max_results=max_results)
    return [page['href'] for page in results]


def web_scrape(url: str) -> str:
    try:
        headers = {
            'User-Agent': 'Chrome/124.0.0.0',
            'Accept-Language': 'en-Us, en'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        else:
            return f'Could not retrieve the webpage: {response.status_code}'
    except Exception as ex:
        print(ex)
        return f'Could not retrieve the webpage: {ex}'


if __name__ == '__main__':
    question = 'Which national parks are in California?'

    llm = mistral()

    assistant_selection_prompt = get_prompt_template('assistant-selection').format(user_question = question)
    assistant_instructions = llm.invoke(assistant_selection_prompt)
    assistant_instructions_dict = to_obj(assistant_instructions.content)

    web_search_prompt = get_prompt_template('web-search').format(
        assistant_instructions = assistant_instructions_dict['assistant_instructions'],
        num_search_queries = 2,
        user_question = assistant_instructions_dict['user_question']
    )
    web_search_queries = llm.invoke(web_search_prompt)
    web_search_queries_list = to_obj(web_search_queries.content.replace('\n', ''))

    searches_and_result_urls = []
    for wq in web_search_queries_list:
        query = wq['search_query']
        searches_and_result_urls.append({
            'search_query': query,
            'result_urls': web_search(query, max_results=3)
        })

    search_query_and_result_url_list = []
    for qr in searches_and_result_urls:
        search_query_and_result_url_list.extend([{'search_query': qr['search_query'], 'result_url': r} for r in qr['result_urls']])

    result_text_list = []
    for re in search_query_and_result_url_list:
        result_text = web_scrape(re['result_url'])
        result_text_list.append({
            'result_text': result_text[:10000],
            'result_url': re['result_url'],
            'search_query': re['search_query']
        })

    result_text_summary_list = []
    for rt in result_text_list:
        summary_prompt = get_prompt_template('summary-instructions').format(
            search_result_text=rt['result_text'],
            search_query=rt['search_query']
        )
        text_summary = llm.invoke(summary_prompt)

        result_text_summary_list.append({
            'text_summary': text_summary,
            'result_url': rt['result_url'],
            'search_query': rt['search_query']
        })

    stringified_summary_list = [
        f'Source URL: {summary["result_url"]}\nSummary: {summary["text_summary"]}'
        for summary in result_text_summary_list
    ]

    appended_result_summaries = '\n'.join(stringified_summary_list)

    research_report_prompt = get_prompt_template('research-report').format(
        research_summary = appended_result_summaries,
        user_question = question
    )
    research_report = llm.invoke(research_report_prompt)

    print(research_report.content)