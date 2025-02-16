# Alkatto

[![CI](https://github.com/naoufal51/alkatto/actions/workflows/unit-tests.yml/badge.svg)](https://github.com/naoufal51/alkatto/actions/workflows/unit-tests.yml)
[![Integration Tests](https://github.com/naoufal51/alkatto/actions/workflows/integration-tests.yml/badge.svg)](https://github.com/naoufal51/alkatto/actions/workflows/integration-tests.yml)

A sophisticated LangGraph-based project implementing multiple specialized conversation agents for different purposes. Built with [LangGraph](https://github.com/langchain-ai/langgraph), this project showcases advanced agent architectures for specific use cases.

## Project Components

### 1. Analyst Graph
Located in `src/agent/analyst_graph/`, this component implements an analytical conversation agent designed for data analysis and interpretation tasks.

### 2. Interview Graph
Found in `src/agent/interview_graph/`, this component provides a specialized conversation flow for conducting interviews and gathering structured information.

### 3. Shared Infrastructure
The project includes a robust shared infrastructure (`src/shared/`) with:
- Common configuration management
- Retrieval utilities
- State management
- Utility functions

## Project Structure

```
src/
├── agent/
│   ├── analyst_graph/     # Analyst conversation agent
│   ├── interview_graph/   # Interview conversation agent
│   ├── configuration.py   # Agent configuration
│   ├── graph.py          # Core graph implementation
│   └── state.py          # State management
└── shared/               # Shared utilities and infrastructure
```

## Getting Started

1. Create a `.env` file:
```bash
cp .env.example .env
```

2. Configure your environment variables in `.env`

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Development

The project uses a modular architecture with separate graphs for different use cases:

- **Analyst Graph**: Customize this component when working with analytical conversations and data interpretation tasks
- **Interview Graph**: Modify this component to adjust interview flows and information gathering processes

Each graph component contains its own:
- Configuration (`configuration.py`)
- Graph implementation (`graph.py`)
- State management (`state.py`)
- Custom prompts (`prompts.py`)

## Testing

The project includes both unit and integration tests:
- Unit tests in `tests/unit_tests/`
- Integration tests in `tests/integration_tests/`

Run tests using:
```bash
make test
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the terms of the LICENSE file included in the repository.