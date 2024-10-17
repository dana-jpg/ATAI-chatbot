import numpy as np
from rdflib import Graph
from transformers import T5ForConditionalGeneration, T5Tokenizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time

# File paths for graph data and embeddings
graph_path = '../data/14_graph.nt'
entity_emb_path = '../data/ddis-graph-embeddings/entity_embeds.npy'
relation_emb_path = '../data/ddis-graph-embeddings/relation_embeds.npy'
entity_ids_path = '../data/ddis-graph-embeddings/entity_ids.del'
relation_ids_path = '../data/ddis-graph-embeddings/relation_ids.del'

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2
bot_name = "serene-ocean"
bot_pass = "C7qlH0L2"

# 1. Define the NLU Class
class NLU:
    def __init__(self):
        # Load Hugging Face T5 model for SPARQL query generation
        self.t5_tokenizer = T5Tokenizer.from_pretrained('t5-base')
        self.t5_model = T5ForConditionalGeneration.from_pretrained('t5-base')

        # Load Sentence-BERT for entity matching
        self.bert_model = SentenceTransformer('all-mpnet-base-v2')

        # Load RDF graph and embeddings
        self.graph = Graph()
        self.graph.parse(graph_path, format='turtle')

        # Load entity and relation embeddings
        self.entity_emb = np.load(entity_emb_path)
        self.relation_emb = np.load(relation_emb_path)

        # Updated entity and relation loading with correct parsing
        with open(entity_ids_path, 'r') as f:
            self.ent2id = {}
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    index, entity_id = parts
                    try:
                        self.ent2id[entity_id] = int(index)
                    except ValueError:
                        print(f"Skipping line due to invalid integer value: {line}")
                else:
                    print(f"Skipping line due to incorrect format: {line}")

        with open(relation_ids_path, 'r') as f:
            self.rel2id = {}
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) == 2:
                    index, relation_id = parts
                    try:
                        self.rel2id[relation_id] = int(index)
                    except ValueError:
                        print(f"Skipping line due to invalid integer value: {line}")
                else:
                    print(f"Skipping line due to incorrect format: {line}")


        self.id2ent = {v: k for k, v in self.ent2id.items()}
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        # Precompute entity embeddings from graph
        self.graph_entities = self.load_entity_labels(self.graph)
        self.entity_embeddings = self.bert_model.encode(self.graph_entities, convert_to_tensor=True)

    def load_entity_labels(self, graph):
        """Extracts all entity labels from the graph for embedding comparison."""
        return [str(label) for _, label in graph.subject_objects(predicate=self.graph.namespace_manager.store.namespace("rdfs:label"))]

    def query_to_sparql(self, query):
        """Converts a natural language query to SPARQL using T5."""
        input_text = f"translate English to SPARQL: {query}"
        input_ids = self.t5_tokenizer.encode(input_text, return_tensors='pt')
        output = self.t5_model.generate(input_ids)
        return self.t5_tokenizer.decode(output[0], skip_special_tokens=True)

    def find_closest_entity(self, user_input):
        """Finds the closest matching entity in the graph based on the user query."""
        query_embedding = self.bert_model.encode(user_input, convert_to_tensor=True)
        similarities = util.pytorch_cos_sim(query_embedding, self.entity_embeddings)
        best_match_idx = np.argmax(similarities.cpu().numpy())
        return self.graph_entities[best_match_idx]

    def find_best_relation(self, user_query):
        """Finds the best matching relation using cosine similarity between the query and relation embeddings."""
        query_embedding = self.bert_model.encode(user_query, convert_to_tensor=True)
        similarities = cosine_similarity([query_embedding], self.relation_emb)
        best_relation_idx = np.argmax(similarities)
        return self.id2rel[best_relation_idx]

    def execute_sparql_query(self, sparql_query):
        """Executes the SPARQL query against the RDF graph."""
        results = self.graph.query(sparql_query)
        return results


# 2. Define the Agent Class
class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        
        # Initialize NLU class
        self.NLU = NLU()

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    room.post_messages(f'Hello! This is a welcome message from {room.my_alias}.')
                    room.initiated = True

                # Retrieve messages from this chat room.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new message #{message.ordinal}: '{message.message}' - {self.get_time()}")

                    user_message = str(message.message)
                    print("Message received: " + user_message)

                    # Step 1: Convert user query to SPARQL using the T5 model
                    sparql_query = self.NLU.query_to_sparql(user_message)
                    print(f"Generated SPARQL Query: {sparql_query}")

                    # Step 2: Find the closest entity using Sentence-BERT
                    closest_entity = self.NLU.find_closest_entity(user_message)
                    print(f"Closest entity found: {closest_entity}")

                    # Step 3: Find the best relation using relation embeddings
                    best_relation = self.NLU.find_best_relation(user_message)
                    print(f"Best matching relation: {best_relation}")

                    # Step 4: Execute SPARQL query in the RDF graph
                    results = self.NLU.execute_sparql_query(sparql_query)
                    response = "\n".join([str(result) for result in results])
                    
                    if results:
                        print(f"Query results: {response}")
                        room.post_messages(response)
                    else:
                        room.post_messages(f"Could not find an answer to your query: {user_message}")

                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                for reaction in room.get_reactions(only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new reaction #{reaction.message_ordinal}: '{reaction.type}' - {self.get_time()}")
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


# Run the bot
if __name__ == '__main__':
    demo_bot = Agent(bot_name, bot_pass)
    print("Agent is listening.")
    demo_bot.listen()
