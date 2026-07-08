import json
import operator

from typing import TypedDict, Optional, Annotated
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command
from llm import Mistral
from tools import WebSearchTool, WebScraperTool


class Search(TypedDict):
    query: str
    url: str


class ResearchState(TypedDict):
    user_question: str
    assistant_type: str
    assistant_instructions: str
    search_queries: list[str]
    search_urls: list[Search]
    summaries: Annotated[list[str], operator.add]
    research_report: str


class PipelineWorkerState(TypedDict):
    search_query: str
    search_url: str
    search_result_text: Optional[str]
    summary_text: Optional[str]


SELECT_ASSISTANT = 'select_assistant'
GENERATE_QUERIES = 'generate_queries'
GENERATE_URLS = 'generate_urls'
PIPELINE_WORKER = 'pipeline_worker'
WEB_SCRAPE = 'web_scrape'
SUMMARIZE_SEARCH_RESULT = 'summarize_search_result'
REDUCE_SUMMARIES = 'reduce_summaries'
WRITE_RESEARCH_REPORT = 'write_research_report'


class Worker:

    llm: BaseChatModel

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def prompt_template(self, name: str) -> PromptTemplate:
        with open(f'resources/prompts/{name}.txt') as file:
            text = file.read()
            return PromptTemplate.from_template(text)

    def parse_dict(self, string: str) -> dict:
        try:
            if string.startswith('```json') and string.endswith('```'):
                string = string.replace('```json', '').replace('```', '')
            return json.loads(string)
        except Exception as ex:
            print(ex)
            return {}


class PipelineWorker(Worker):

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.web_scraper_tool = WebScraperTool()
        self.app = self.create_graph()

    def create_graph(self):
        graph = StateGraph(PipelineWorkerState)
        graph.add_node(WEB_SCRAPE, self.web_scrape)
        graph.add_node(SUMMARIZE_SEARCH_RESULT, self.summarize_search_result)
        graph.add_edge(WEB_SCRAPE, SUMMARIZE_SEARCH_RESULT)
        graph.add_edge(SUMMARIZE_SEARCH_RESULT, END)
        graph.set_entry_point(WEB_SCRAPE)
        return graph.compile()

    def web_scrape(self, state: PipelineWorkerState) -> dict | Command:
        url = state['search_url']
        print(f'* Scrapping web page "{url}"')
        try:
            text = self.web_scraper_tool.scrape(url)[:10000]
            return {'search_result_text': text}
        except:
            print(f'* Exception during scrapping web page: {url}')
            return Command(goto=END)

    def summarize_search_result(self, state: PipelineWorkerState) -> dict:
        chain = (
            self.prompt_template('summary-instructions')
            | self.llm
            | StrOutputParser()
        )
        print(f'* Summarizing search result: {state["search_result_text"][:50]}...')
        summary = chain.invoke(state)
        return {'summary_text': summary}

    def invoke(self, state: PipelineWorkerState) -> PipelineWorkerState:
        return self.app.invoke(state)


class ResearchAgent(Worker):

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.search_queries_count = 2
        self.search_tool = WebSearchTool(max_results=2)
        self.pipeline_worker = PipelineWorker(llm)
        self.app = self.create_graph()

    def create_graph(self):
        graph = StateGraph(ResearchState)
        graph.set_entry_point(SELECT_ASSISTANT)
        # nodes
        graph.add_node(SELECT_ASSISTANT, self.select_assistant)
        graph.add_node(GENERATE_QUERIES, self.generate_search_queries)
        graph.add_node(GENERATE_URLS, self.generate_search_urls)
        graph.add_node(PIPELINE_WORKER, self.call_pipeline_worker)
        graph.add_node(WRITE_RESEARCH_REPORT, self.write_research_report)
        # edges
        graph.add_edge(SELECT_ASSISTANT, GENERATE_QUERIES)
        graph.add_edge(GENERATE_QUERIES, GENERATE_URLS)
        graph.add_conditional_edges(GENERATE_URLS,self.split_pipeline_work,[PIPELINE_WORKER])
        graph.add_edge(PIPELINE_WORKER, WRITE_RESEARCH_REPORT)
        graph.add_edge(WRITE_RESEARCH_REPORT, END)
        return graph.compile()

    def select_assistant(self, state: ResearchState) -> dict:
        chain = (
            self.prompt_template('assistant-selection')
            | self.llm
            | StrOutputParser()
            | self.parse_dict
        )
        assistant_info = chain.invoke(state)
        print(f'* Selected assistant: {assistant_info["assistant_type"]}')
        return {
            'assistant_type': assistant_info['assistant_type'],
            'assistant_instructions': assistant_info['assistant_instructions']
        }

    def generate_search_queries(self, state: ResearchState) -> dict:
        chain = (
            RunnableLambda(lambda x: x | {'num_search_queries': self.search_queries_count})
            | self.prompt_template('web-search')
            | self.llm
            | StrOutputParser()
            | RunnableLambda(lambda result_text: result_text.split('\n'))
        )
        search_queries = [query for query in chain.invoke(state) if query.strip()]
        for query in search_queries:
            print(f'* Generated query: {query}')
        return {'search_queries': search_queries}

    def generate_search_urls(self, state: ResearchState) -> dict:
        queries = state['search_queries']
        searches = []
        for query in queries:
            for url in self.search_tool.links(query):
                searches.append({'query': query, 'url': url})
        return {'search_urls': searches}

    def split_pipeline_work(self, state: ResearchState) -> list[Send]:
        searches = state['search_urls']
        return [Send(PIPELINE_WORKER, {'search_query': search['query'], 'search_url': search['url']}) for search in searches]

    def call_pipeline_worker(self, state: PipelineWorkerState) -> dict:
        result = self.pipeline_worker.invoke(state)
        if 'summary_text' not in result:
            return {'summaries': []}
        source_url = result['search_url']
        summary_text = result['summary_text']
        return {'summaries': [f'Source URL: {source_url}\nSummary: {summary_text}']}

    def write_research_report(self, state: ResearchState) -> dict:
        user_question = state['user_question']
        summaries = '\n'.join(state['summaries'])
        chain =(
            RunnableLambda(lambda x: {
                'research_summary': summaries,
                'user_question': user_question,
            })
            | self.prompt_template('research-report')
            | self.llm
            | StrOutputParser()
        )
        report = chain.invoke(state)
        return {'research_report': report}

    def process(self, user_question: str) -> str:
        state = self.app.invoke({'user_question': user_question})
        return state['research_report']


if __name__ == '__main__':
    question = 'Which national parks are in California?'
    llm = Mistral()
    research_agent = ResearchAgent(llm)
    answer = research_agent.process(question)
    print(answer)
