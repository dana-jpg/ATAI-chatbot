# graph loading
import pickle
from rdflib import Graph

from speakeasypy import Speakeasy, Chatroom
from typing import List
import time

# imports from dataset_intro
import rdflib
from rdflib.namespace import Namespace, RDF, RDFS, XSD
from rdflib.term import URIRef, Literal
import csv
import json
import networkx as nx
import pandas as pd
import rdflib
from collections import defaultdict, Counter
import re # for regular expressions
import numpy as np
import os
from sklearn.metrics import pairwise_distances
import editdistance

# nlp
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import spacy
from spacy.matcher import Matcher

import pandas as pd
import spacy
import pandas as pd
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# file paths
graph_path = '../data/14_graph.nt'
entity_emb_path = '../data/ddis-graph-embeddings/entity_embeds.npy'
relation_emb_path = '../data/ddis-graph-embeddings/relation_embeds.npy'
entity_ids_path = '../data/ddis-graph-embeddings/entity_ids.del'
relation_ids_path = '../data/ddis-graph-embeddings/relation_ids.del'

header = '''
    PREFIX ddis: <http://ddis.ch/atai/>
    PREFIX wd: <http://www.wikidata.org/entity/>
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    PREFIX schema: <http://schema.org/>
'''


# import lookup table
word_pred = pd.read_csv("../data/word_pred.csv")
pred_labels = list(word_pred['Label'])
pred_keys = list(word_pred['Key'])


#pred_keys = list(word_pred.iloc[:,2])
print("Predicate Lookup Table Loaded!")

# convert lookup data to dict
pred_dict = {}
for i in range(0,len(pred_labels)):
    pred_dict[pred_labels[i]] = pred_keys[i]


DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2
bot_name = "serene-ocean"
bot_pass = "C7qlH0L2"

test_query = '''
    PREFIX ddis: <http://ddis.ch/atai/> 
    PREFIX wd: <http://www.wikidata.org/entity/> 
    PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
    PREFIX schema: <http://schema.org/>
    
    SELECT ?lbl WHERE {
        SELECT ?movie ?lbl ?rating WHERE {
            ?movie wdt:P31 wd:Q11424 .
            ?movie ddis:rating ?rating .
            ?movie rdfs:label ?lbl .
        }
        ORDER BY ASC(?rating) 
        LIMIT 10
    }
    '''

class SPARQLUtility:
    
    @staticmethod
    def format_query(raw_query: str) -> str:
        """Formats a raw SPARQL query string for execution."""
        formatted_query = ' '.join(raw_query.split())
        formatted_query = formatted_query.replace("   ", "\t")
        return formatted_query

class KnowledgeGraph():
    def __init__(self, graph_path, entity_emb_path, relation_emb_path, entity_ids_path, relation_ids_path, format):
        # defining prefixes for the graph
        self.WD = Namespace('http://www.wikidata.org/entity/')
        self.WDT = Namespace('http://www.wikidata.org/prop/direct/')
        self.SCHEMA = Namespace('http://schema.org/')
        self.DDIS = Namespace('http://ddis.ch/atai/')
        
        self.data = Graph()  # Initialize an empty RDFLib graph
        try:
            self.data.parse(graph_path, format="turtle")  # Parse the graph using N-Triples format
            print("Loading graph completed!")
        except Exception as e:
            print(f"Failed to load the graph: {e}")
            exit(1)

        self.entities = set(self.data.subjects()) | {s for s in self.data.objects() if isinstance(s, URIRef)}
        self.predicates = set(self.data.predicates())
        self.literals = {s for s in self.data.objects() if isinstance(s, Literal)}

        # list of all literals
        self.literals_list = [str(literal) for literal in self.literals]

        # list of all subject, label, literal relations: contains all labeled subjects
        self.tuple_list = list(self.data.query(header + '''
                                SELECT ?s ?lbl WHERE {
                                    '''#?s ?p wd:Q11424 .
                                    '''?s rdfs:label ?lbl
                                }
                                '''))

        print("Loading graph embeddinings ...")
        self.entity_emb = np.load(entity_emb_path)
        self.relation_emb = np.load(relation_emb_path)
        print("Loading graph embeddinings completed!")

        print("Loading dictionaries")
        # load the dictionaries
        with open(entity_ids_path, 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}

        with open(relation_ids_path, 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.data.subject_objects(RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        print("Dictionaries loaded!")
    
    def sparql_query(self, query):
        results = self.data.query(query)
        return [str(s) for s, in results]
    
    # match_subject_literal > only returns one Qid
    def match_lit(self, input):
        tmp = 9999
        match_node = ""
        entity = input
        # print("--- entity matching for \"{}\"\n".format(entity))
        for key, value in self.tuple_list:
            # print("edit distance between {} and {}: {}".format(value, entity, editdistance.eval(value, entity)))
            if editdistance.eval(value, entity) < tmp:
                tmp = editdistance.eval(value, entity)
                match_subject = key
                match_literal = value
                # returns matched literal and subject key
        return match_literal, match_subject
    
    # finds closest literal in graph data compared to spec. word
    def find_closest_lit(self, word, return_min_dist=False):
        min_distance = float('inf')
        closest_literal = None

        for literal in self.literals_list:
            distance = editdistance.eval(word, literal)

            if distance < min_distance:
                min_distance = distance
                closest_literal = literal

        if return_min_dist:
            return closest_literal, min_distance
        else:
            return closest_literal


    # returns a list of all predicates/relations to one entity
    def request_relations_to_entity(self, entity):
        q = header + f'''
        SELECT DISTINCT ?predicate WHERE {{
        ?subject ?predicate <http://www.wikidata.org/entity/{entity}> .
    }}
'''
        predicates = self.data.query(q)
        return self.URIRef_to_key(predicates)




    # Returns list of URIRef objects into list of entity keys (e.g Q3811376, ...)
    def URIRef_to_key(self, tuple_list):
        print("Tuple list is: " + str(tuple_list))
        entity_ids = []
        for tuple_item in tuple_list:
            print("Tuple item is: " + str(tuple_item))
            uri_str = str(tuple_item[0])
            entity_id = uri_str.split('/')[-1]
            entity_ids.append(entity_id)
        return entity_ids
        
    # you can query the key of an entity by given its entity literal or entity literal and according predicate key – entity label is literal
    def query_entity_keys(self, entity_label, pred_key = None):
        if pred_key == None:
            tuple_list = list(self.data.query(header + '''
                SELECT ?entity WHERE {
                ?entity rdfs:label "'''+entity_label +'''"@en .
                }
            '''))
        return self.URIRef_to_key(tuple_list)
        
   
        
    # entity_id is in form Q849901, rel_id in form P57 
    def find_embedding(self, entity_id, rel_id):
        entity_URIRef = "http://www.wikidata.org/entity/" + str(entity_id)
        rel_URIRef = "http://www.wikidata.org/prop/direct/" + str(rel_id)
        entity_URIRef = URIRef(entity_URIRef)
        rel_URIRef = URIRef(rel_URIRef)
        entity_result = self.entity_emb[self.ent2id[entity_URIRef]]
        relation_result = self.relation_emb[self.rel2id[rel_URIRef]]
        # Apply TransE Scoring Function
        lhs = entity_result + relation_result
        # Calculate distance
        dist = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1)
        # Sort entities based on distance
        most_likely = dist.argsort()
        # Select most likely result from list
        # show most likely entities
       
        dataframe = pd.DataFrame([
        (self.id2ent[idx][len(self.WD):], self.ent2lbl[self.id2ent[idx]], dist[idx], rank+1)
        for rank, idx in enumerate(most_likely[:10])],
        columns=('Entity', 'Label', 'Score', 'Rank'))

        # Extract the first row
        first_row = dataframe.iloc[0]
        
        return first_row

class NLU(): # Natural Language Understanding Unit
    def __init__(self):
        
        

        self.BERT_tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
        self.BERT_model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
        self.BERT_nlp = pipeline("ner", model=self.BERT_model, tokenizer=self.BERT_tokenizer)

        self.Spacy_nlp = spacy.load('en_core_web_sm')
    

    def filter_by_predicate(self, predicate_list):
        filtered_data = word_pred[word_pred['predicate'].isin(predicate_list)]
        return filtered_data


    # This will be done, using spacy nlp:
    def tokenize_sentence(self, sentence_string):
        # Process the sentence
        doc = self.Spacy_nlp(sentence_string)
        # Print out tokens:
        for token in doc:
            print(token.text)
    
    # adjacent predicates can be called by relations_to_entity
    def predicate_similarities(self, sentence, entity_id, adjacent_predicates, filter_predicates = True):
        doc = self.Spacy_nlp(sentence)
        results = defaultdict(list)
        """
        print("The adjacent predicates are: ")
        for i in adjacent_predicates:
            print(i)
        """
        #if filter_predicates:
        #    word_pred = self.filter_by_predicate(adjacent_predicates)

        for i, token in enumerate(doc):
            for n_gram_size in range(1, 2):  # 1-gram to 3-gram
                if i + n_gram_size <= len(doc):
                    n_gram_text = ' '.join([doc[j].text for j in range(i, i + n_gram_size)])
                    n_gram_vector = self.Spacy_nlp(n_gram_text).vector.reshape(1, -1)  # Reshape for cosine_similarity

                    # Calculate similarity
                    similarities = [
                        (word, predicate, cosine_similarity(n_gram_vector, self.Spacy_nlp(word).vector.reshape(1, -1))[0][0])
                        for word, predicate in zip(word_pred['Label'], word_pred['Key'])
                    ]

                    # Sort by similarity (descending) and keep top 5
                    top_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]
                    results[n_gram_text].extend(top_similarities)

        # Convert to DataFrame for better visualization
        output = []
        for n_gram, similar_words in results.items():
            for rank, (word, predicate, similarity) in enumerate(similar_words, 1):
                output.append((n_gram, rank, word, predicate, similarity))

        result_df = pd.DataFrame(output, columns=['n-gram', 'rank', 'word', 'predicate', 'similarity'])
        sorted_result = result_df.sort_values(by='similarity', ascending=False)

        # calculate average similarity of each pred:
        pred_table = {}
        for i in range(0, sorted_result.shape[0]):
            pred = sorted_result.iloc[i]['predicate']
            similarity = sorted_result.iloc[i]['similarity']

            if pred in pred_table:
                pred_table[pred].append(similarity)
            else:
                pred_table[pred] = [similarity]

        average_dict = {}
        for pred, values in pred_table.items():
            if values:
                average_value = sum(values) / len(values)
                average_dict[pred] = average_value
            else:
                average_dict[pred] = 0
        
        sorted_average_dict = {k: v for k, v in sorted(average_dict.items(), key=lambda item: item[1], reverse=True)}

        return sorted_average_dict


    # Named Entitiy Recognition
    def find_named_entities(self, sentence):
        return self.BERT_nlp(sentence)
    
    # For deriving predicate relations
    def find_relations(self, entity_string, sentence):
      doc = self.Spacy_nlp(sentence)

      token = None
      for ent in doc.ents:
          if ent.text == entity_string:
              token = ent.root
              break
          
      predicate_token = None
      if token:
          for ancestor in token.ancestors:
              if ancestor.pos_ == 'VERB':  # suchen nach einem Verb
                  predicate_token = ancestor
                  break

      # return predicate
      if predicate_token:
          return predicate_token.text
      else:
          return None

class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        # load graph
        self.movie_graph = KnowledgeGraph(graph_path= graph_path, entity_emb_path = entity_emb_path, relation_emb_path=relation_emb_path, entity_ids_path=entity_ids_path, relation_ids_path=relation_ids_path, format="turtle")
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
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    # Print user message in the console:

                    print("Message received: "+str(message.message))
                    user_message = str(message.message)

                    print("NLU is searching for entities ...")
                    entities = self.NLU.find_named_entities(user_message)
                    print("NER completed!")
                    if entities:
                        entity = "" 
                        for ent in entities:
                            entity += " " + ent['word'].replace("#", "")
                        print("NLU found entity: " + str(entity))
    
                        # get closest entity
                        closest_lit = self.movie_graph.find_closest_lit(entity)
                        print("Closest literal is: " + str(closest_lit))
                        
                        # get entity ids
                        ids = self.movie_graph.query_entity_keys(closest_lit)
                        print("IDS are: " + str(ids))
                        if len(ids) == 0:
                            room.post_messages("Can you repeat your question more clearly?")    
                        best_answers = []
                        adjacent_predicates = []
                        id = 0
                        predicates = self.NLU.predicate_similarities(user_message, id, adjacent_predicates, False)
                        print("Predicates are: " + str(predicates))
                        for predicate_id, value in predicates.items():
                            print(f"Predicate is: {predicate_id} with value: {value}")

                        for id in ids:                             
                            
                            for predicate_id, value in predicates.items():
                                print(f"Predicate is: {predicate_id} with value: {value}")
                                rel_URIRef = "http://www.wikidata.org/prop/direct/" + str(predicate_id)
                                rel_URIRef = URIRef(rel_URIRef)
                                cnt = 0
                                try:
                                    if self.movie_graph.rel2id[rel_URIRef]:
                                        #print("Best possible predicate is: " + str(predicate_id))
                                        answer = self.movie_graph.find_embedding(str(id), str(predicate_id))
                                        print(answer)
                                        cnt += 1
                                        if str(answer['Label']) != str(closest_lit):
                                            #print("Label is: " + str(answer['Label']) + " and closest literal is: " + str(closest_lit))
                                            best_answers.append(answer)
                                            print("appended answer: " + answer)
                                        if cnt == 6:
                                             break
                                except KeyError:
                                    print("Not possible to find this predicate")
                            ####################################

                            #answer = self.movie_graph.find_embedding(str(id), str(predicate_id))
                            #print(answer)
                            #best_answers.append(answer)
                        
                        # get the best result
                        min = 9999
                        min_label = "Embeddings did not find the suitable answer"
                        print("Best answers are: " + str(best_answers))
                        for answer in best_answers:
                            print("One of answers is: " + str(answer))
                            if answer['Score'] < min:
                                min = answer['Score']
                                min_label = answer['Label']
                        
                        print("Best answer is: " + str(min_label))
                        room.post_messages(min_label)

                    else:
                        room.post_messages("I didnt' understand? What did you want to say?")

                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent(bot_name, bot_pass)
    print("Agent is listening.")
    demo_bot.listen()

"""
if __name__ == '__main__':
    ref = (rdflib.term.Literal('Harry Potter', lang='en'), rdflib.term.URIRef('http://www.wikidata.org/entity/Q3244512'))
    uri = ref[1]
    print(uri)
"""
