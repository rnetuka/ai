import logging
import operator

from typing import TypedDict, Optional, Annotated, cast
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser, ListOutputParser, NumberedListOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langgraph.errors import NodeError
from langgraph.graph import StateGraph, END
from langgraph.types import Send, Command
from llm import Mistral
from tools import WebSearchTool, WebScraperTool
from tools.web_scraper import WebScraperException


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
logger.addHandler(console_handler)


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
    summary_exceptions: Annotated[list[Exception], operator.add]
    exception: Optional[Exception]
    research_report: str


class PipelineWorkerState(TypedDict):
    search_query: str
    search_url: str
    search_result_text: Optional[str]
    summary_text: Optional[str]
    exception: Optional[Exception]


SELECT_ASSISTANT = 'select_assistant'
GENERATE_QUERIES = 'generate_queries'
GENERATE_URLS = 'generate_urls'
PIPELINE_WORKER = 'pipeline_worker'
WEB_SCRAPE = 'web_scrape'
SUMMARIZE_SEARCH_RESULT = 'summarize_search_result'
REDUCE_SUMMARIES = 'reduce_summaries'
WRITE_RESEARCH_REPORT = 'write_research_report'


class PipelineWorkerException(Exception): pass
class ResearchAgentException(Exception): pass


class Worker:

    llm: BaseChatModel

    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def prompt_template(self, name: str) -> PromptTemplate:
        with open(f'resources/prompts/{name}.txt') as file:
            text = file.read()
            return PromptTemplate.from_template(text)


class PipelineWorker(Worker):

    def __init__(self, llm: BaseChatModel):
        super().__init__(llm)
        self.web_scraper_tool = WebScraperTool()
        self.app = self.create_graph()

    def create_graph(self):
        graph = StateGraph(PipelineWorkerState)
        graph.add_node(WEB_SCRAPE, self.web_scrape, error_handler=self.handle_web_scrape_exception)
        graph.add_node(SUMMARIZE_SEARCH_RESULT, self.summarize_search_result)
        graph.add_edge(WEB_SCRAPE, SUMMARIZE_SEARCH_RESULT)
        graph.add_edge(SUMMARIZE_SEARCH_RESULT, END)
        graph.set_entry_point(WEB_SCRAPE)
        return graph.compile()

    @property
    def summary_instructions(self) -> PromptTemplate:
        return self.prompt_template('summary-instructions')

    def web_scrape(self, state: PipelineWorkerState) -> dict:
        url = state['search_url']
        try:
            logger.info(f'Scrapping web page {url}')
            text = self.web_scraper_tool.scrape(url)[:10000]
            return {'search_result_text': text}
        except WebScraperException as ex:
            logger.error(f'Exception during scrapping web page: {url}')
            raise ex

    def handle_web_scrape_exception(self, _: PipelineWorkerState, error: NodeError) -> Command:
        return Command(update={'exception': error.error}, goto=END)

    def summarize_search_result(self, state: PipelineWorkerState) -> dict:
        logger.info(f'Summarizing search result: {state["search_result_text"][:50]}... for query: {state["search_query"]}')
        chain = self.summary_instructions | self.llm | StrOutputParser()
        summary = chain.invoke(state)
        return {'summary_text': summary}

    def invoke(self, state: PipelineWorkerState) -> PipelineWorkerState:
        result = self.app.invoke(state)
        if 'exception' in result:
            raise PipelineWorkerException() from result['exception']

        return cast(PipelineWorkerState, result)





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
        graph.add_node(WRITE_RESEARCH_REPORT, self.write_research_report, error_handler=self.handle_research_exception)
        # edges
        graph.add_edge(SELECT_ASSISTANT, GENERATE_QUERIES)
        graph.add_edge(GENERATE_QUERIES, GENERATE_URLS)
        graph.add_conditional_edges(GENERATE_URLS,self.split_pipeline_work,[PIPELINE_WORKER])
        graph.add_edge(PIPELINE_WORKER, WRITE_RESEARCH_REPORT)
        graph.add_edge(WRITE_RESEARCH_REPORT, END)
        return graph.compile()

    @property
    def select_assistant_instructions(self) -> PromptTemplate:
        return self.prompt_template('assistant-selection')

    @property
    def web_search_instructions(self) -> PromptTemplate:
        return self.prompt_template('web-search')

    @property
    def research_report_instructions(self) -> PromptTemplate:
        return self.prompt_template('research-report')

    @property
    def exception_description_instructions(self) -> PromptTemplate:
        return self.prompt_template('exception')

    def select_assistant(self, state: ResearchState) -> dict:
        chain = self.select_assistant_instructions | self.llm | JsonOutputParser()
        assistant_info = chain.invoke(state)
        logger.info(f'Selected assistant: {assistant_info["assistant_type"]}')
        return assistant_info

    def generate_search_queries(self, state: ResearchState) -> dict:
        chain = self.web_search_instructions | self.llm | NumberedListOutputParser()
        search_queries = chain.invoke(state | {'num_search_queries': self.search_queries_count})

        for query in search_queries:
            logger.info(f'Generated query: {query}')

        return {'search_queries': search_queries}

    def generate_search_urls(self, state: ResearchState) -> dict:
        queries = state['search_queries']
        searches = [{'query': query, 'url': url} for query in queries for url in self.search_tool.links(query)]
        return {'search_urls': searches}

    def split_pipeline_work(self, state: ResearchState) -> list[Send]:
        searches = state['search_urls']
        return [Send(PIPELINE_WORKER, {'search_query': search['query'], 'search_url': search['url']}) for search in searches]

    def call_pipeline_worker(self, state: PipelineWorkerState) -> dict:
        try:
            result = self.pipeline_worker.invoke(state)
            source_url = result['search_url']
            summary_text = result['summary_text']
            summary = f'Source URL: {source_url}\nSummary: {summary_text}'
            return {'summaries': [summary]}
        except PipelineWorkerException as ex:
            return {'summary_exceptions': [ex]}

    def write_research_report(self, state: ResearchState) -> dict:
        summaries = state['summaries']

        if len(summaries) == 0:
            raise ResearchAgentException('No summaries has been provided')

        summaries = '\n'.join(summaries)
        chain = self.research_report_instructions | self.llm | StrOutputParser()
        report = chain.invoke(state | {'research_summary': summaries})
        return {'research_report': report}

    def handle_research_exception(self, state: ResearchState, error: NodeError) -> Command:
        exceptions = [error.error] + state['summary_exceptions']
        exceptions = '\n'.join(str(ex) for ex in exceptions)
        chain = self.exception_description_instructions | self.llm | StrOutputParser()
        description = chain.invoke({'exceptions': exceptions})
        return Command(
            update={'exception': description},
            goto=END
        )

    def process(self, user_question: str) -> str:
        state = self.app.invoke({'user_question': user_question})
        return state['exception'] if 'exception' in state else state['research_report']


if __name__ == '__main__':
    question = 'Which national parks are in California?'
    llm = Mistral()
    research_agent = ResearchAgent(llm)
    answer = research_agent.process(question)
    print(answer)
