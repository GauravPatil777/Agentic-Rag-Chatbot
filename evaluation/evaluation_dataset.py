from deepeval.test_case import LLMTestCase
from backend.chatbot_backend import pool

golden_dataset=[
  {
    "input": "How have AI agents evolved over the last ten years?",
    "expected_output": "Over the last decade, AI has shifted from rigid, rule-based systems and narrow task automation toward sophisticated autonomous agent architectures powered by large language models. Modern agents can plan, execute, and adapt across workflows with less continuous human oversight.",
    "category": "overview"
  },
  {
    "input": "What were the main limitations of traditional AI systems described in the report?",
    "expected_output": "Traditional systems relied on brittle, deterministic rule trees and procedural scripts, and were restricted to narrow, specialized tasks.",
    "category": "factual"
  },
  {
    "input": "What capabilities make modern LLM-driven agents different from static response engines?",
    "expected_output": "Modern LLM-driven agents can act as task executors: they plan, execute, and adapt across end-to-end workflows, such as project scoping and execution, with minimal human intervention.",
    "category": "conceptual"
  },
  {
    "input": "Which three architectural capabilities are highlighted as important for agent autonomy?",
    "expected_output": "The report highlights reasoning frameworks, memory management, and tool-use capabilities as important architectural components for agent autonomy.",
    "category": "factual"
  },
  {
    "input": "How do Chain-of-Thought and self-reflection loops help AI agents?",
    "expected_output": "The report says these reasoning techniques enable models to break down complex goals and iterate on their outputs.",
    "category": "conceptual"
  },
  {
    "input": "How do agents maintain continuity across extended interactions?",
    "expected_output": "The report describes memory management using vector retrieval and external storage to address context-retention challenges and maintain coherent, personalized interactions over extended periods.",
    "category": "factual"
  },
  {
    "input": "What role does Agentic RAG play in memory and reasoning?",
    "expected_output": "The report gives Agentic RAG as an example in which retrieval is treated as part of an iterative reasoning toolkit.",
    "category": "conceptual"
  },
  {
    "input": "Which agent frameworks are named in the section about tool use?",
    "expected_output": "The report names AutoGen, CrewAI, LangGraph, and Kore.ai as frameworks associated with capabilities for invoking external software APIs, executing code, and dynamically calling tools within feedback loops.",
    "category": "factual"
  },
  {
    "input": "What is a multi-agent system, according to the report?",
    "expected_output": "The report describes multi-agent systems as coordinated systems in which specialized agents collaborate within complex workflows. Agents can delegate subtasks, simulate social environments, and cooperatively resolve complex objectives.",
    "category": "conceptual"
  },
  {
    "input": "Which multi-agent frameworks or systems are mentioned?",
    "expected_output": "The report mentions MASFactory, MASFly, and Langroid. It says these use graph-based orchestration, dynamic adaptation, and human-in-the-loop interactions to improve reliability and flexibility.",
    "category": "factual"
  },
  {
    "input": "What weaknesses does the report identify in current AI-agent evaluation benchmarks?",
    "expected_output": "The report says many current benchmarks focus heavily on static question-answering tasks rather than dynamic, multi-round real-world environments. It also identifies a need for unified benchmarks that assess memory, learning, and multi-agent interactions together.",
    "category": "evaluation"
  },
  {
    "input": "What does the report say is needed to improve evaluation of AI agents?",
    "expected_output": "It identifies a need for unified, comprehensive benchmarks covering memory, learning, and multi-agent interactions, as well as dynamic, multi-round evaluation in realistic environments.",
    "category": "evaluation"
  },
  {
    "input": "What deployment challenges can prevent AI-agent projects from reaching enterprise-wide impact?",
    "expected_output": "The report identifies operational governance gaps, difficulty handling unstructured inputs, and failures to recover gracefully from tool errors as challenges that can lead to project abandonment before enterprise-wide impact.",
    "category": "deployment"
  },
  {
    "input": "Which domain is described as the fastest-growing agent role segment, and what growth is projected?",
    "expected_output": "Software engineering is described as the fastest-growing agent role segment, with a projected compound annual growth rate of over 50% from 2025 to 2030.",
    "category": "factual"
  },
  {
    "input": "What are the five key insights listed in the report?",
    "expected_output": "The five key insights are: (1) the paradigm shift from deterministic scripts and rule-based systems to adaptive LLM-driven agents; (2) architecture as a differentiator through memory, iterative reasoning, and tool calling; (3) multi-agent synergy through specialized task distribution and workflow coordination; (4) the adoption-scaling gap involving governance, orchestration, and scaling challenges; and (5) benchmarking lag, because dynamic evaluation of memory, learning, and multi-agent synergy remains an active research area.",
    "category": "multi_fact"
  },
  {
    "input": "What does the report conclude about the future challenge for enterprise transformation?",
    "expected_output": "The report concludes that moving from isolated experimental deployments to scalable enterprise transformation requires overcoming challenges in operational governance, dynamic evaluation, and architectural reliability.",
    "category": "conclusion"
  },
  {
    "input": "Does the report identify the exact names of companies that have successfully deployed AI agents at enterprise scale?",
    "expected_output": "The report does not identify specific companies that have successfully deployed AI agents at enterprise scale. It discusses enterprise adoption and deployment challenges in general.",
    "category": "unanswerable"
  },
  {
    "input": "What is the name of the author who wrote the original research paper on Chain-of-Thought mentioned in the report?",
    "expected_output": "The report mentions Chain-of-Thought as a reasoning technique but does not identify the author or authors of the original research paper.",
    "category": "unanswerable"
  }
]

def fetch_user_interactions():

    with pool.connection() as conn:
        with conn.cursor() as cursor:

            cursor.execute("""
                SELECT user_input, ai_response, retrieval_context
                FROM evaluation_interactions
                ORDER BY created_at DESC
            """)

            rows = cursor.fetchall()

    test_cases = []
    i=0
    for row in rows:

        test_case = LLMTestCase(
            input=row[0],
            actual_output=row[1],
            expected_output=golden_dataset[i]["expected_output"],
            retrieval_context=row[2] or []
        )
        i+=1
        test_cases.append(test_case)

    return test_cases


test_cases = fetch_user_interactions()