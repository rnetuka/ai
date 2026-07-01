import json
import requests
from bs4 import BeautifulSoup
from typing import List, Any, TypeVar, NamedTuple, overload
from ddgs import DDGS
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from llm import Mistral


T = TypeVar('T', bound=dict | list[Any])


def parse(string: str, expected_type: type[T]) -> T:
    try:
        if string.startswith('```json') and string.endswith('```'):
            string = string.replace('```json', '').replace('```', '')
        return json.loads(string)
    except Exception:
        return {}


Assistant = NamedTuple('Assistant', [('type', str), ('instructions', str)])
SearchResult = NamedTuple('SearchResult', [('query', str), ('url', str), ('text', str)])
Summary = NamedTuple('Summary', [('text', str), ('url', str)])


class WebSearchTool:

    max_results: int

    def __init__(self):
        self.max_results = 3

    def links(self, query: str) -> list[str]:
        results = DDGS().text(query, max_results=self.max_results)
        return [page['href'] for page in results]


class WebScraperTool:

    def scrape(self, url: str) -> str:
        headers = {
            'User-Agent': 'Chrome/124.0.0.0',
            'Accept-Language': 'en-Us, en'
        }
        response = requests.get(url, headers=headers, timeout=15)
        if response.ok:
            soup = BeautifulSoup(response.text, 'html.parser')
            return soup.get_text(separator=' ', strip=True)
        else:
            raise Exception(f'Could not retrieve the webpage: {response.status_code}')



class Research:

    llm: BaseChatModel

    def __init__(self):
        self.llm = Mistral()
        self.search_tool = WebSearchTool()
        self.web_scraper_tool = WebScraperTool()

    def prompt_template(self, name: str) -> PromptTemplate:
        with open(f'resources/prompts/{name}.txt') as file:
            text = file.read()
            return PromptTemplate.from_template(text)

    def select_assistant_prompt(self, question: str) -> str:
        return self.prompt_template('assistant-selection').format(user_question = question)

    def select_assistant(self, question: str) -> Assistant:
        response = self.llm.invoke(self.select_assistant_prompt(question))
        assistant = parse(response.content, dict)
        return Assistant(type = assistant['assistant_type'], instructions = assistant['assistant_instructions'])

    def web_search_prompt(self, assistant: str, question: str) -> str:
        return self.prompt_template('web-search').format(
            assistant_instructions = assistant,
            user_question = question,
            num_search_queries = 2
        )

    def create_search_queries(self, assistant: Assistant, question: str) -> list[str]:
        prompt = self.web_search_prompt(assistant.instructions, question)
        response = self.llm.invoke(prompt)
        queries = response.content.split('\n')
        return queries

    def search(self, query: str) -> list[SearchResult]:
        results = []
        for url in self.search_tool.links(query):
            try:
                result = self.web_scraper_tool.scrape(url)[:10000]
                results.append(SearchResult(query, url, result))
            except Exception as ex:
                print(f'Exception during web scraping: {ex}')
        return results

    def summary_prompt(self, text: str, question: str) -> str:
        return self.prompt_template('summary-instructions').format(search_result_text=text, search_query=question)

    def summarize(self, text: str, question: str) -> str:
        prompt = self.summary_prompt(text, question)
        response = self.llm.invoke(prompt)
        return response.content

    def summarize_result(self, search_result: SearchResult) -> Summary:
        text = self.summarize(search_result.text, search_result.query)
        return Summary(text, search_result.url)

    def research_report_prompt(self, question: str, summaries: str) -> str:
        return self.prompt_template('research-report').format(
            research_summary=summaries,
            user_question=question
        )

    def write_report(self, question: str, summaries: list[Summary]) -> str:
        prompt = self.research_report_prompt(
            question=question,
            summaries='\n'.join(f'Source URL: {summary.url}\nSummary: {summary.text}' for summary in summaries)
        )
        research_report = self.llm.invoke(prompt)
        return research_report.content

    def process(self, question: str) -> str:
        assistant = self.select_assistant(question)
        print(f'Selected assistant: {assistant.type}')
        print('---')
        queries = self.create_search_queries(assistant, question)
        print('\n'.join(f'Query {i + 1}: {query}' for i, query in enumerate(queries)))
        print('---')
        results = [result for query in queries for result in self.search(query)]
        summaries = [self.summarize_result(result) for result in results]
        return self.write_report(question, summaries)


if __name__ == '__main__':
    question = 'Which national parks are in California?'
    research = Research()
    answer = research.process(question)
    print(answer)
