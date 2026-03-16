# Show Neptune graph schema

Display the complete Neptune Analytics graph schema for ContextWeave.

Read and summarize these files:
- `src/shared/models.py` — NodeType and EdgeType enums
- `docs/graph_schema.md` — Full schema specification
- `src/preprocessor/graph_builder.py` — How nodes and edges are constructed

Then display:

## Node Types
List all node labels, their properties, and purpose.

## Edge Types
List all edge labels, source → target types, edge properties, and purpose.

## Routing Intelligence Graph
Show the EFFECTIVE_FOR subgraph structure used for adaptive routing:
- RetrievalStrategy nodes (graph_first, hybrid, keyword_boosted, semantic_search)
- QuestionType nodes (skill_depth, architecture, comparison, project, credential, general)
- EFFECTIVE_FOR edges with weight property (0.10 – 1.00)

## Example Queries
Show 3-5 useful openCypher queries for exploring the graph.

Keep the output concise — use tables where possible.
