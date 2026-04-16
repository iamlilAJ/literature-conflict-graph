# RAG Conflict Demo

This is a small real-paper demo produced from OpenAlex metadata plus LLM claim
extraction over retrieval-augmented generation papers.

Open `index.html` in a browser to explore the graph.

The demo contains:

- 5 real paper records with OpenAlex links.
- 9 extracted claims.
- 39 graph nodes and 50 graph edges.
- 1 detected benchmark inconsistency around RAG on domain QA.
- 5 possible explanations and follow-up checks.

To keep the repository lightweight and redistribution-friendly, `papers.jsonl`
keeps paper titles, years, venues, and links, but omits full abstracts and paper
text. Claims include short evidence spans for inspection.

