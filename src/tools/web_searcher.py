from ddgs import DDGS


class WebSearchTool:

    max_results: int

    def __init__(self, max_results=3):
        self.max_results = max_results

    def links(self, query: str) -> list[str]:
        results = DDGS().text(query, max_results=self.max_results)
        return [page['href'] for page in results]
