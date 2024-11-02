import sys
# import os
print(sys.path)
sys.path.append('/Users/ninar/opt/anaconda3/envs/speakeasy-env/lib/python3.9/site-packages')

import numpy as np
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
import spacy
from scipy.spatial.distance import cosine
from transformers import pipeline
import json
from rdflib import URIRef, Literal
import os
import numpy as np
from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph, Namespace
from rdflib.namespace import RDF
import spacy
from scipy.spatial.distance import cosine
from transformers import pipeline
import json
from rdflib import URIRef, Literal
import rdflib
from sklearn.metrics import pairwise_distances
import csv
import editdistance



############################## LOADING ######################################################


# Load spaCy NLP model for English
nlp = spacy.load("en_core_web_md")  # Use medium model for larger vectors

# Load a Hugging Face transformer model for NER
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

# defining prefixes
WD = rdflib.Namespace('http://www.wikidata.org/entity/')
WDT = rdflib.Namespace('http://www.wikidata.org/prop/direct/')
DDIS = rdflib.Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = rdflib.Namespace('http://schema.org/')


# Define a namespace for your knowledge graph
EX = Namespace("http://example.org/")

# Load the RDF knowledge graph using N-Triples format
print("Loading RDF knowledge graph...")
g = Graph()
try:
    g.parse(source="../../2nd_Year-1st_Semester/ATAI/Dataset/14_graph.nt", format="turtle")  # Replace with actual file and format
    print("Knowledge graph loaded successfully.")
except Exception as e:
    print(f"Failed to load the knowledge graph: {e}")
    exit(1)

print("Loading the embeddings...")
entity_embeddings = np.load(os.path.join('data', '/Users/ninar/Desktop/University of Zurich/Classes/2nd_Year-1st_Semester/ATAI/Dataset/ddis-graph-embeddings/entity_embeds.npy'))
relation_embeddings = np.load(os.path.join('data', '/Users/ninar/Desktop/University of Zurich/Classes/2nd_Year-1st_Semester/ATAI/Dataset/ddis-graph-embeddings/relation_embeds.npy'))
entity_ids = os.path.join('data', '/Users/ninar/Desktop/University of Zurich/Classes/2nd_Year-1st_Semester/ATAI/Dataset/ddis-graph-embeddings/entity_ids.del')
relation_ids = os.path.join('data', '/Users/ninar/Desktop/University of Zurich/Classes/2nd_Year-1st_Semester/ATAI/Dataset/ddis-graph-embeddings/relation_ids.del')


# load the dictionaries
with open(entity_ids, 'r') as ifile:
    ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
    id2ent = {v: k for k, v in ent2id.items()}
with open(relation_ids, 'r') as ifile:
    rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
    id2rel = {v: k for k, v in rel2id.items()}

ent2lbl = {ent: str(lbl) for ent, lbl in g.subject_objects(RDFS.label)}
lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

rel2lbl = {rel: str(lbl) for rel, lbl in g.subject_objects(RDFS.label) if rel in rel2id}
lbl2rel = {lbl: rel for rel, lbl in rel2lbl.items()}

# Load relation labels
with open('ATAI-chatbot/speakeasy-python-client-library/usecases/relation_labels.json', 'r', encoding='utf-8') as f:
    relation_id_to_label = json.load(f)

# Create mapping from label to property ID (e.g., 'color' -> 'P462')
relation_label_to_id = {}
for uri, label in relation_id_to_label.items():
    prop_id = uri.split('/')[-1]  # Extract 'P462' from the URI
    relation_label_to_id[label] = prop_id




############################## CHECK - UNDERTSANDING THE DICTIONARIES ######################################################


# # Print the first 10 items from each dictionary
# print("First 10 items in ent2id:")
# for k, v in list(ent2id.items())[:10]:
#     print(f"{k}: {v}")

# print("\nFirst 10 items in id2ent:")
# for k, v in list(id2ent.items())[:10]:
#     print(f"{k}: {v}")

# print("\nFirst 10 items in rel2id:")
# for k, v in list(rel2id.items())[:10]:
#     print(f"{k}: {v}")

# print("\nFirst 10 items in id2rel:")
# for k, v in list(id2rel.items())[:10]:
#     print(f"{k}: {v}")

# print("\nFirst 10 items in ent2lbl:")
# for k, v in list(ent2lbl.items())[:10]:
#     print(f"{k}: {v}")

# print("\nFirst 10 items in lbl2ent:")
# for k, v in list(lbl2ent.items())[:10]:
#     print(f"{k}: {v}")

# print("First 10 items in relation_label_to_id:")
# for label, prop_id in list(relation_label_to_id.items())[:10]:
#     print(f"{label}: {prop_id}")

# # print("\nFirst 10 items in entity_label_to_id:")
# # for label, ent_id in list(entity_label_to_id.items())[:10]:
# #     print(f"{label}: {ent_id}")

# # Print the first 10 items from each dictionary
# print("First 10 items in rel2lbl:")
# for rel, lbl in list(rel2lbl.items())[:10]:
#     print(f"{rel}: {lbl}")

# print("\nFirst 10 items in lbl2rel:")
# for lbl, rel in list(lbl2rel.items())[:10]:
#     print(f"{lbl}: {rel}")



# print("TESTING FUNCTION get_pid_from_ent2lbl") # TOP za P

def get_pid_from_ent2lbl(label):
    # Look up the label in the reversed ent2lbl (lbl2ent)
    for uri, lbl in ent2lbl.items():
        if lbl == label:
            return uri.split('/')[-1]  # Extract P... or Q... part if it exists
    return None  # Return None if label not found

# # Example Tests
# print("Test cases for get_pid_from_ent2lbl:")
# print("ID for 'genre' P136:", get_pid_from_ent2lbl("genre"))  # Expected: P161
# print("ID for 'cast member' P161:", get_pid_from_ent2lbl("cast member"))  # Expected: P161
# print("ID for 'MPAA film rating' P1657:", get_pid_from_ent2lbl("MPAA film rating"))  # Expected: P1657
# print("ID for 'Netflix ID' P1874:", get_pid_from_ent2lbl("Netflix ID"))  # Expected: P1874
# print("ID for 'Rotten Tomatoes ID' P1258:", get_pid_from_ent2lbl("Rotten Tomatoes ID"))  # Expected: P1258

# print("ID for 'cast member' P161:", get_pid_from_ent2lbl("cast member"))  # Expected: P161
# print("ID for 'MPAA film rating' P1657:", get_pid_from_ent2lbl("MPAA film rating"))  # Expected: P1657
# print("ID for 'Netflix ID' P1874:", get_pid_from_ent2lbl("Netflix ID"))  # Expected: P1874
# print("ID for 'Rotten Tomatoes ID' P1258:", get_pid_from_ent2lbl("Rotten Tomatoes ID"))  # Expected: P1258
# print("ID for 'genre' P136:", get_pid_from_ent2lbl("genre"))  # Expected: P136
# print("ID for 'director' P57:", get_pid_from_ent2lbl("director"))  # Expected: P57
# print("ID for 'country of origin' P495:", get_pid_from_ent2lbl("country of origin"))  # Expected: P495



#############################################################################################


# Improved function to get the closest entity QID given a label
def get_best_matching_entity_qid(label):
    closest_distance = float('inf')
    best_match_uri = None
    for lbl, uri in lbl2ent.items():
        distance = editdistance.eval(label, lbl)
        if distance < closest_distance:
            closest_distance = distance
            best_match_uri = uri
    if best_match_uri:
        return best_match_uri.split('/')[-1]  # Extract the Q... part
    return None

# print("Entity Tests get_best_matching_entity_qid(label):")
# print("ID for 'Finding Nemo' Q132863:", get_best_matching_entity_qid("Finding Nemo"))  # Expected: Q132863
# print("ID for 'Finding Dory' Q9321426:", get_best_matching_entity_qid("Finding Dory"))  # Expected: Q9321426
# print("ID for 'Up' Q174811:", get_best_matching_entity_qid("Up"))  # Expected: Q174811
# print("ID for 'WALL·E' Q104905:", get_best_matching_entity_qid("WALL·E"))  # Expected: Q104905
# print("ID for 'Toy Story 2' Q187266:", get_best_matching_entity_qid("Toy Story 2"))  # Expected: Q187266
# print("ID for 'Cars 2' Q192212:", get_best_matching_entity_qid("Cars 2"))  # Expected: Q192212
# print("ID for 'Cars' Q182153:", get_best_matching_entity_qid("Cars"))  # Expected: Q182153


def qp_given_words(e, l):

    # q = get_entity_qid_from_label(e)
    # p = get_pid_from_ent2lbl(l)
    q = get_best_matching_entity_qid(e)
    p = get_pid_from_ent2lbl(l)
    return q, p

# def embedding_result_given_qp(q, p):
def embedding_result_given_qp(e, l):

    q, p = qp_given_words(e, l)
    print("q, p extracted from embedding_result_given_qp(e, l)")
    print(q,p)

    # Find the Wikidata ID for the movie (https://www.wikidata.org/wiki/Q132863 is the ID for "Finding Nemo")
    movie = WD[q]

    # Find the movie in the graph
    movie_id = ent2id[movie]

    # we compare the embedding of the query entity to all other entity embeddings
    distances = pairwise_distances(entity_embeddings[movie_id].reshape(1, -1), entity_embeddings, metric='cosine').flatten()

    # and sort them by distance
    most_likely = distances.argsort()

    # we print rank, entity ID, entity label, and distance
    for rank, idx in enumerate(most_likely[:20]):
        rank = rank + 1
        ent = id2ent[idx]
        q_id = ent.split('/')[-1]
        lbl = ent2lbl[ent]
        dist = distances[idx]

        print(f'{rank:2d}. {dist:.3f} {q_id:10s} {lbl}')


    movie_emb = entity_embeddings[ent2id[movie]]

    # Find the predicate (relation) of the genre (https://www.wikidata.org/wiki/Property:P136 is the ID for "genre")
    genre = WDT[p]
    genre_emb = relation_embeddings[rel2id[genre]]

    # combine according to the TransE scoring function
    lhs = movie_emb + genre_emb

    # compute distance to *any* entity
    distances = pairwise_distances(lhs.reshape(1, -1), entity_embeddings).reshape(-1)

    # find most plausible tails
    most_likely = distances.argsort()

    # show most likely entities
    for rank, idx in enumerate(most_likely[:20]):
        rank = rank + 1
        ent = id2ent[idx]
        q_id = ent.split('/')[-1]
        lbl = ent2lbl[ent]
        dist = distances[idx]

        print(f'{rank:2d}. {dist:.3f} {q_id:10s} {lbl}')



embedding_result_given_qp('Toy Story 2', 'genre')
