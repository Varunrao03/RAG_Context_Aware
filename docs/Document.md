# Cosine Similarity Vs Eucledian Distance

-> For both the Raw Vector Search and Mocked Vertex based search I have used Cosine similarity,
the reason is that all the embeddings are l2 normalised before storage and before querying.

-> why I chose Cosine over Eucledian ?
There are some metrics I considered base on some research i did:

a. **Scale invariance** = in COsine it doesn't affect the score whereas it affects the score in eucledian.
b. **Senetence Embeddings** = In Cosine it is a Natural fir i.e. models are trained to encode meaning as direction, 
                          Whereas in Eucledian embeddings magnitude carries no semantic meaning.
c. **Measures** = Cosiine measures angle between vectors whereas Eucledian measures absolute dist. in space.

Also, sentence transformer models are trained trained with contrastive objectives that encode semantic 
similarity as the angle between vectors, not their magnitude. Two sentences with identical meaning but
different lengths will produce vectors pointing in the same direction but potentially with different magnitudes. 
Cosine similarity correctly scores them as identical while Euclidean distance would not.

# Migrating to Vertex AI Vector Seach.
We willl be using Vertex AI Vector Search only when we have the follwoing conditions:

a. **Large Corpora**: When your dataset exceeds ~100,000 chunks, which is typically the limit for single-machine RAM.
b. **High Performance**: When you need low latency (~10ms) that remains stable even as the corpus grows to billions of vectors.
c. **Production Reliability**: When you need a managed service that handles cost-efficient index storage and high-speed approximate nearest neighbor (ANN) searches.

## To migrate we have to follow some changes like:

*Step 1*. We will have to replace the embedding.py and retrieval.py with the actual Vertex AI SDK initialization.
**Step 2*. Initialize the Tree-AH index and deploy it to a public endpoint.
*Step 3*. Create a VertexVectorStore class in rag/vertex_store.py. This acts as a drop-in replacement that implements 
        the same .add() and .search() interface as your local store but talks to the Vertex endpoint.
*Step 4*. Update the main retrieval class to inject the new VertexVectorStore and re-run the ingestion script to populate the cloud index.
