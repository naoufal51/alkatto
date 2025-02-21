"""Prompts for the research analysis subgraph."""

# Research focus prompts
FOCUS_INSTRUCTIONS = """
Based on the research question, identify the main research focus.
Break down the query into:
1. Core research area
- Primary field of study
- Key theoretical concepts
- Main research problems

2. Specific topics or methodologies
- Research methods
- Technical approaches
- Analytical frameworks
- Experimental designs

3. Relevant academic fields
- Primary disciplines
- Related fields
- Interdisciplinary connections
- Specialized subfields

4. Time period of interest
- Historical context
- Current developments
- Future directions
- Temporal constraints

5. Key concepts to explore
- Fundamental theories
- Technical terminology
- Core principles
- Essential frameworks

Focus on academic and scientific aspects of the query.
Identify both theoretical and practical research elements.
Consider cross-disciplinary implications and connections."""

# Query generation prompts
QUERY_INSTRUCTIONS = """
Generate an arXiv search query based on the research question.
The query should:

1. Use appropriate arXiv categories and fields
- Include relevant primary categories
- Consider cross-listed categories
- Use field-specific qualifiers
- Apply category-specific filters

2. Include relevant technical terminology
- Use standardized terminology
- Include common variations
- Consider acronyms and abbreviations
- Add relevant synonyms

3. Focus on specific research aspects
- Target precise concepts
- Include methodological terms
- Specify technical components
- Add relevant metrics

4. Consider temporal relevance
- Use date range constraints
- Include version information
- Consider submission vs. update dates
- Account for field evolution

5. Use boolean operators effectively
- AND for required terms
- OR for alternatives
- NOT for exclusions
- Parentheses for grouping

Example transformations:
- "Latest developments in transformer models" -> 
  "cat:cs.CL AND (transformer OR attention) AND title:model AND submittedDate:[2023 TO 2024]"
- "Quantum computing error correction" -> 
  "cat:quant-ph AND (error correction OR quantum error) AND abstract:implementation"
- "Deep learning optimization techniques" ->
  "(cat:cs.LG OR cat:stat.ML) AND (optimization OR training) AND (gradient OR loss)"

Structure queries to maximize relevance while maintaining sufficient breadth."""

# Analysis prompts
ANALYSIS_INSTRUCTIONS = """Analyze research papers considering:

1. Key findings and contributions
- Novel theoretical contributions
  * New concepts or theories
  * Mathematical frameworks
  * Algorithmic innovations
- Technical innovations
  * Implementation advances
  * System architectures
  * Performance improvements
- Empirical results
  * Experimental outcomes
  * Statistical analyses
  * Benchmark comparisons
- Methodological advances
  * Novel approaches
  * Improved techniques
  * Enhanced frameworks

2. Methodology and approach
- Research design
  * Experimental setup
  * Control measures
  * Variable selection
- Data collection and analysis
  * Dataset characteristics
  * Preprocessing methods
  * Analysis techniques
- Experimental setup
  * Hardware configuration
  * Software environment
  * Parameter settings
- Validation methods
  * Evaluation metrics
  * Statistical tests
  * Robustness checks

3. Technical innovations
- Novel algorithms or methods
  * Algorithmic improvements
  * Computational efficiency
  * Theoretical guarantees
- Architectural improvements
  * System design
  * Component integration
  * Scalability features
- Performance optimizations
  * Speed enhancements
  * Resource efficiency
  * Quality improvements
- Implementation details
  * Code structure
  * Technical requirements
  * Deployment considerations

4. Experimental results
- Quantitative metrics
  * Performance measures
  * Efficiency metrics
  * Quality indicators
- Comparative analysis
  * Baseline comparisons
  * Ablation studies
  * Alternative approaches
- Statistical significance
  * Confidence intervals
  * P-values
  * Effect sizes
- Benchmark performance
  * Standard datasets
  * Common metrics
  * Industry benchmarks

5. Limitations and future work
- Current constraints
  * Technical limitations
  * Resource constraints
  * Dataset limitations
- Methodological limitations
  * Approach restrictions
  * Validation gaps
  * Generalization issues
- Open challenges
  * Unsolved problems
  * Technical barriers
  * Research gaps
- Future research directions
  * Potential improvements
  * New applications
  * Extended capabilities

6. Impact on the field
- Theoretical implications
  * Conceptual advances
  * Framework extensions
  * Knowledge contributions
- Practical applications
  * Industry relevance
  * Implementation potential
  * Real-world impact
- Industry relevance
  * Commercial applications
  * Market potential
  * Business impact
- Research community impact
  * Citation potential
  * Research direction
  * Community influence

Provide a comprehensive analysis that:
- Maintains academic rigor
- Highlights key technical details
- Identifies research trends
- Evaluates scientific merit
- Assesses reproducibility
- Considers practical implications
- Notes theoretical advances
- Examines methodological strength"""

# Summary prompts
SUMMARY_INSTRUCTIONS = """Generate a research summary that:

1. Synthesizes key findings from all papers
- Identify common themes
  * Shared concepts
  * Similar approaches
  * Related findings
- Highlight major discoveries
  * Breakthrough results
  * Novel methods
  * Significant advances
- Note conflicting results
  * Contradictory findings
  * Methodological differences
  * Varying conditions
- Summarize consensus views
  * Agreed principles
  * Common conclusions
  * Shared understanding

2. Identifies common themes and patterns
- Methodological trends
  * Popular approaches
  * Emerging techniques
  * Declining methods
- Shared challenges
  * Common obstacles
  * Technical barriers
  * Resource limitations
- Common assumptions
  * Theoretical bases
  * Standard practices
  * Shared frameworks
- Recurring limitations
  * Persistent issues
  * Common constraints
  * Typical problems

3. Highlights methodological innovations
- Novel approaches
  * New techniques
  * Original methods
  * Innovative solutions
- Technical breakthroughs
  * Performance advances
  * Efficiency gains
  * Quality improvements
- Experimental designs
  * Novel setups
  * Improved controls
  * Better validation
- Analytical frameworks
  * New perspectives
  * Enhanced analysis
  * Better understanding

4. Discusses practical implications
- Industry applications
  * Commercial potential
  * Implementation needs
  * Market impact
- Implementation considerations
  * Technical requirements
  * Resource needs
  * Deployment challenges
- Scalability aspects
  * Growth potential
  * Performance scaling
  * Resource scaling
- Resource requirements
  * Hardware needs
  * Software dependencies
  * Human expertise

5. Suggests future research directions
- Open problems
  * Unsolved challenges
  * Research gaps
  * Future needs
- Promising approaches
  * Potential solutions
  * New methods
  * Emerging techniques
- Potential improvements
  * Enhancement areas
  * Optimization opportunities
  * Quality advances
- Research opportunities
  * New directions
  * Collaboration potential
  * Funding prospects

6. Maintains academic rigor and clarity
- Use precise terminology
  * Technical accuracy
  * Clear definitions
  * Consistent usage
- Cite specific results
  * Data points
  * Measurements
  * Statistical findings
- Maintain objectivity
  * Balanced view
  * Evidence-based
  * Unbiased analysis
- Acknowledge limitations
  * Known constraints
  * Data gaps
  * Methodological issues

Structure the summary with:
- Clear section headers
- Logical flow of ideas
- Technical accuracy
- Accessible language

Focus on making complex research accessible while preserving technical accuracy and academic rigor.
Use appropriate citations and references throughout the summary."""

# Confidence scoring prompts
CONFIDENCE_SCORING_INSTRUCTIONS = """Score research confidence based on:

1. Paper quality and impact
- Journal/conference reputation
  * Publication venue ranking
  * Peer review process
  * Impact factor
- Citation count
  * Total citations
  * Citation rate
  * Citation quality
- Author expertise
  * Academic credentials
  * Research history
  * Field recognition
- Peer review status
  * Review process
  * Reviewer expertise
  * Review thoroughness

2. Methodology robustness
- Research design
  * Experimental setup
  * Control measures
  * Variable selection
- Data quality
  * Dataset size
  * Data cleanliness
  * Representation
- Statistical validity
  * Statistical methods
  * Significance levels
  * Effect sizes
- Reproducibility
  * Method description
  * Code availability
  * Data access

3. Result reliability
- Statistical significance
  * P-values
  * Confidence intervals
  * Effect sizes
- Sample size
  * Dataset scale
  * Population coverage
  * Statistical power
- Control measures
  * Experimental controls
  * Bias mitigation
  * Error handling
- Validation methods
  * Cross-validation
  * Independent verification
  * Robustness checks

4. Technical depth
- Mathematical rigor
  * Theoretical foundation
  * Mathematical proofs
  * Formal analysis
- Implementation details
  * Code quality
  * System architecture
  * Technical specifications
- Experimental setup
  * Hardware configuration
  * Software environment
  * Parameter settings
- Comparative analysis
  * Baseline comparisons
  * Alternative methods
  * Ablation studies

5. Temporal relevance
- Publication date
  * Time since publication
  * Field evolution rate
  * Current relevance
- Citation freshness
  * Recent citations
  * Citation trends
  * Impact longevity
- Technology currency
  * Technical obsolescence
  * State-of-the-art comparison
  * Implementation viability
- Field evolution
  * Research progress
  * Paradigm shifts
  * New developments

Provide a score between 0.0 and 1.0 with detailed justification for each component.
Consider both quantitative metrics and qualitative aspects in the scoring.
Weight factors based on research field and paper type.
Document any assumptions or limitations in the confidence assessment."""