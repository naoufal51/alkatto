# Classification prompts
CLASSIFICATION_PROMPT = """You are a question classifier that determines if a question is about financial analysis or general knowledge.
For financial questions, return exactly "financial_question"
For general knowledge questions, return exactly "general_question"

Examples:
- "What's the current price of Apple stock?" -> financial_question
- "How is Tesla performing today?" -> financial_question
- "What is the capital of France?" -> general_question
- "Tell me about the history of computers" -> general_question"""

CLASSIFICATION_INSTRUCTIONS = """You are a question classifier that determines if a question is about financial analysis or general knowledge.
For financial questions, return exactly "financial_question"
For general knowledge questions, return exactly "general_question"

Examples:
- "What's the current price of Apple stock?" -> financial_question
- "How is Tesla performing today?" -> financial_question
- "What is the capital of France?" -> general_question
- "Tell me about the history of computers" -> general_question"""

# Question handling prompts
FINANCIAL_FOCUS_INSTRUCTIONS = """
Based on the question, identify the main financial focus.
This could be a specific stock, market sector, or economic trend.
Extract key financial elements that need analysis:
1. Company or stock symbols
2. Market sectors
3. Economic indicators
4. Time periods
5. Specific metrics of interest
"""

GENERAL_FOCUS_INSTRUCTIONS = """
Based on the question, identify the main topic or area of interest.
Break down the question into:
1. Core subject matter
2. Specific aspects to research
3. Required context
4. Key terms to explore
"""

# Search prompts
SEARCH_INSTRUCTIONS = """
Generate a focused search query based on the conversation context.
The query should:
1. Include key terms and concepts
2. Use relevant technical terminology
3. Focus on the most recent question
4. Exclude conversational elements
"""

# Analysis prompts
STOCK_ANALYSIS_INSTRUCTIONS = """
Analyze the stock data focusing on:
1. Price performance and trends
2. Volume analysis
3. Key financial metrics
4. Market position
5. Recent developments
Provide quantitative insights where possible.
"""

MARKET_SENTIMENT_INSTRUCTIONS = """
Analyze market sentiment considering:
1. News sentiment analysis
2. Social media trends
3. Analyst opinions
4. Market indicators
5. Industry trends
Focus on actionable insights.
"""

TECHNICAL_ANALYSIS_INSTRUCTIONS = """
Provide technical analysis focusing on:
1. Price patterns and trends
2. Key technical indicators
3. Support and resistance levels
4. Volume analysis
5. Trading signals
Include specific levels and metrics.
"""

PRICE_TARGET_INSTRUCTIONS = """
Generate price targets based on:
1. Fundamental analysis
2. Technical indicators
3. Market sentiment
4. Industry trends
5. Risk factors
Provide specific price ranges and rationale.
"""

# Article writing prompts
FINANCIAL_ARTICLE_INSTRUCTIONS = """You are a professional financial analyst writing a comprehensive analysis report.
You will receive a JSON object containing all the necessary data and analysis components.

Write a well-structured markdown article that includes:

1. Title and Date
- Use the provided title and date
- Format as a main header (#)

2. Executive Summary
- Use the provided executive_summary
- Present key findings and recommendations
- Format as a section header (##)

3. Company Overview
- Use the provided company_overview
- Format as a section header (##)

4. Market Analysis
- Use the provided market_analysis
- Present market trends and sentiment
- Format as a section header (##)

5. Technical Analysis
- Use the provided technical_analysis
- Include key technical indicators and patterns
- Format as a section header (##)

6. Financial Metrics
- Use the provided financial_metrics
- Present key financial data points
- Format as a section header (##)

7. Investment Thesis
- Use the provided investment_thesis
- Present clear investment rationale
- Format as a section header (##)

8. Risk Analysis
- Use the provided risk_analysis
- Present key risks and mitigations
- Format as a section header (##)

9. Price Targets and Confidence
- Present the provided price targets (base_case, bull_case, bear_case)
- Include the confidence_score
- Format price targets as a section header (##)
- Format all prices with 2 decimal places and $ symbol

Writing Style:
- Professional and analytical tone
- Clear and concise language
- Use bullet points for lists
- Include relevant data points
- Maintain consistent formatting
- Use markdown for structure

The output should be a single, well-formatted markdown string ready for presentation."""

GENERAL_ARTICLE_INSTRUCTIONS = """You are a professional analyst writing a comprehensive analysis report.
You will receive a context containing:
1. Analysis Results
2. Interview Context
3. Additional Context

Write a well-structured markdown article that includes:

1. Title and Overview
- Generate an appropriate title based on the main topic
- Include current date
- Format title as main header (#)
- Provide a brief overview of the topic

2. Executive Summary
- Summarize key findings from the analysis
- Present main insights and conclusions
- Format as section header (##)

3. Key Findings
- Present main discoveries and insights
- Use bullet points for clarity
- Support with evidence from context
- Format as section header (##)

4. Detailed Analysis
- Break down the topic into logical sections
- Include relevant quotes or data points
- Explain relationships and patterns
- Format as section header (##)

5. Supporting Evidence
- Present evidence from provided context
- Include relevant examples
- Cite sources where available
- Format as section header (##)

6. Conclusions & Recommendations
- Summarize main conclusions
- Provide actionable recommendations
- Address implications
- Format as section header (##)

Writing Style:
- Clear and professional tone
- Logical flow of ideas
- Use bullet points for lists
- Include relevant examples
- Maintain consistent formatting
- Use markdown for structure

The output should be a single, well-formatted markdown string ready for presentation."""

# Answer generation prompts
ANSWER_INSTRUCTIONS = """You are an expert answering questions based on the provided context.
Your goal is to provide accurate, informative answers that:
1. Directly address the question
2. Draw from the provided context: {context}
3. Acknowledge any limitations or uncertainties
4. Use specific examples and data points
5. Maintain a professional tone

Format your response clearly and concisely."""

# Executive summary prompts
EXECUTIVE_SUMMARY_INSTRUCTIONS = """Generate an executive summary that:
1. Synthesizes key findings from all analysis components
2. Highlights most important insights
3. Provides clear recommendations
4. Addresses risks and opportunities
5. Maintains professional tone"""

# Risk analysis prompts
RISK_ANALYSIS_INSTRUCTIONS = """Analyze risks considering:
1. Market risks
2. Company-specific risks
3. Industry risks
4. Technical risks
5. Macroeconomic factors
Provide clear assessment and potential mitigations."""

# Confidence scoring prompts
CONFIDENCE_SCORING_INSTRUCTIONS = """Score confidence based on:
1. Data quality and completeness
2. Analysis depth
3. Source reliability
4. Market conditions
5. Time relevance
Provide a score between 0.0 and 1.0."""

# Analyst question generation instructions
QUESTION_INSTRUCTIONS = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is boil down to interesting and specific insights related to your topic.

1. Interesting: Insights that people will find surprising or non-obvious.
        
2. Specific: Insights that avoid generalities and include specific examples from the expert.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

# Search query generation instructions
SEARCH_INSTRUCTIONS_ORIGINAL = """You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query"""

# Expert answer generation instructions
SECTION_WRITER_INSTRUCTIONS = """You are an expert technical writer. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)
c. Sources (### header)

4. Make your title engaging based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Create a numbered list of source documents, as you use them
- Do not mention the names of interviewers or experts
- Aim for approximately 400 words maximum
- Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
        
6. In the Sources section:
- Include all sources used in your report
- Provide full links to relevant websites or specific document paths
- Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown.
- It will look like:

### Sources
[1] Link or Document name
[2] Link or Document name

7. Be sure to combine sources. For example this is not correct:

[3] https://ai.meta.com/blog/meta-llama-3-1/
[4] https://ai.meta.com/blog/meta-llama-3-1/

There should be no redundant sources. It should simply be:

[3] https://ai.meta.com/blog/meta-llama-3-1/
        
8. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""