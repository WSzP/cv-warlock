---
allowed-tools: Read, Write, Edit, Bash
argument-hint: [orchestration-type] | --parallel | --sequential | --conditional | --pipeline-optimization
description: Orchestrate comprehensive Tavily optimization with intelligent execution and optimization
model: opus
---

# Tavily Optimizer

Orchestrate intelligent Tavily Optimization **$ARGUMENTS**

## Current Context

Here are Tavily's recommended practices and common patterns to optimize search usage.

1. Use Sensible Defaults for Simple Queries
For quick fact-lookups or basic informational queries, use default / minimal settings: low max_results, default search depth, no raw content extraction. This keeps latency low, results small, and limits overhead for context consumption by LLMs.

2. Adjust Depth / Results Based on Task Complexity
- For research, summarization, or RAG (retrieval-augmented generation) tasks needing deep context: increase search_depth, raise max_results, possibly enable include_raw_content.
- For simple Q&A or single-fact retrieval: stick to basic depth and limited results.

3. Use Domain Filters and Time Filters for Targeted Relevance
- When the query requires trustworthy or authoritative sources, use include_domains (whitelist) to restrict results to known good domains.
- Use exclude_domains to avoid low-quality or untrusted content.
- For time-sensitive queries, apply recency filters (e.g. topic = "news", time_range/days) to prioritize fresh content.

4. Cache and Reuse Results When Possible
If your application or agent may run similar or repeated queries, cache results to avoid redundant API calls. This reduces cost and improves response time, especially for high-frequency queries.

5. Combine Search with Other Tavily Tools Strategically
- Use search for discovery of relevant links/pages.
- If you need deeper content (full article, structured extraction), follow up with extract, crawl, or map where appropriate. This avoids overloading search results while giving richer content only when needed.
- For RAG pipelines: search → extract → summarize/ingest into vector store. This pattern balances cost, depth, and performance.

6. Account for Latency and Rate Limits
- Typical response times are fast, but complex queries (advanced search) may take longer.
- Respect rate limits of your plan. Avoid too many parallel heavy queries without throttling or batching.

7. Use Structured Output - Avoid Parsing Raw HTML in Agent Logic
Tavily returns clean, structured JSON optimized for LLMs and agents, this avoids brittle HTML scraping, reduces errors, and simplifies downstream logic.

Sample Parameter Strategy by Use Case
Use Case	Recommended Config / Strategy
Quick fact lookup / Q&A	Default search settings, max_results = 5, no raw content, basic depth
Up-to-date news / recent events	Use topic = "news", set time_range to "day" or "week", limit results, optionally include raw content for summarization
Research / summarization / RAG	search_depth = advanced, max_results = 10–20, enable include_raw_content, caching + follow-up extraction
Trusted-source retrieval (docs, spec)	Use include_domains whitelist, maybe combine with domain-specific crawls or extract calls
High-volume or batch queries	Cache results, rate-limit, batch queries, reuse previous outputs where possible
Example: Optimized Search Call for a Research Agent
from tavily import TavilyClient

tavily_client = TavilyClient(api_key="YOUR_API_KEY")

response = tavily_client.search(
    query="2025 trends in web-agent search APIs",
    search_depth="advanced",
    max_results=10,
    include_raw_content=True,
    include_images=False,
    include_answer=False,
    exclude_domains=["spammy-site.com","low-quality-blog.org"]
)
This setup balances depth and breadth, fetches full content for analysis, but avoids irrelevant or low-quality sources, keeping results cleaner and more relevant.

Why These Practices Matter for Production Agents
By following these best practices, agents built on Tavily:

- Run faster, giving near real-time responses for simple queries.
- Consume fewer API credits per useful result, cost-efficient at scale.
- Generate cleaner context for LLMs, reduces hallucinations and context noise.
- Scale gracefully, high-volume or batch workloads remain manageable.
- Maintain content quality, using trusted sources and fresh data when necessary.

This turns Tavily Search from a convenient developer tool into a reliable, production-grade building block for AI agents and RAG apps.

## Task

Implement intelligent Tavily execution optimization and resource management:

**Orchestration Type**: Use $ARGUMENTS to focus on Tavily optimization

**Output**: Complete Tavily setup with optimized execution, intelligent resource management, and cost efficiency.