
import csv
import json
import sys
from datetime import datetime
import os
print("sys.path")
print(sys.path)
sys.path.append('/Users/ninar/opt/anaconda3/envs/speakeasy-env/lib/python3.9/site-packages')
sys.path.append('/Users/ninar/Downloads/serene-ocean/speakeasypy/src')
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'speakeasypy', 'src')))
import rdflib
import spacy

import random

from speakeasypy.src.speakeasy import Speakeasy
from speakeasypy.src.chatroom import Chatroom
from speakeasypy.src.question_processor import Question_processor


from typing import List
import time
from transformers import pipeline
import numpy as np
import pandas as pd


DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

BASE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#print(BASE_PATH)

class Agent:
    def __init__(self, username, password):
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()

    def execute_sparql_query(self, query):
        try:
            response = [str(s) for s, in self.graph.query(query)]
            if response:
                res = f"The answer to your question is {response[0]}"
                res = res.encode('ascii', 'xmlcharrefreplace')
                # answer_template = "Hi, the {} of {} is {}".format(relation, entity, answer)
                return res.decode('ascii')
            else:
                return "No answer found for your question."
        except Exception as e:
            print(e)
            return str(e)
        
    def answer_question(self, natural_query):
        if "SELECT" in natural_query.message.upper():
            response = self.execute_sparql_query(natural_query.message)
        else:
            if len(ner_pipeline(natural_query.message, aggregation_strategy="simple")) > 0:
                response = cqp.get_nlp_answers(natural_query.message)
            else:
                response = "Sorry, did not find any matches for your query. Could you please rephrase and try again!"
        return response


    def listen(self):
        while True:
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    room.post_messages('Hello! This is a welcome message from {room.my_alias}. How can I help you?')
                    room.initiated = True
                
                for message in room.get_messages(only_partner=True, only_new=True):
                    print(f"Received message: {message.message}")

                    response = self.answer_question(message)
                    room.post_messages(f"'{response}'")
                    room.mark_as_processed(message)
            
                for reaction in room.get_reactions(only_new=True):
                    print(f"\t- Chatroom {room.room_id} - new reaction #{reaction.message_ordinal}: '{reaction.type}' - {self.get_time()}")
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    g = rdflib.Graph()
    g.parse(os.path.join(BASE_PATH, 'data', '14_graph.nt'), format='turtle')
    RDFS = rdflib.namespace.RDFS
    print("Knowledge graph loaded successfully.")

    ner_pipeline = pipeline('ner', model='planeB/roberta_movie_w_title')
    entity_emb = np.load(os.path.join(BASE_PATH, 'data', 'entity_embeds.npy'))
    relation_emb = np.load(os.path.join(BASE_PATH, 'data', 'relation_embeds.npy'))
    spacy_nlp = spacy.load("en_core_web_sm")

    # Load entity_ids.del and create ent2id and id2ent
    with open(os.path.join(BASE_PATH, 'data', 'entity_ids.del'), 'r') as ifile:
        ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
        id2ent = {v: k for k, v in ent2id.items()}

    # Load relation_ids.del and create rel2id and id2rel
    with open(os.path.join(BASE_PATH, 'data', 'relation_ids.del'), 'r') as ifile:
        rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
        id2rel = {v: k for k, v in rel2id.items()}

    ent2lbl = {ent: str(lbl) for ent, lbl in g.subject_objects(RDFS.label)}
    lbl2ent = {lbl: ent for ent, lbl in ent2lbl.items()}

    # Pass all data to Question_processor
    cqp = Question_processor(g, ner_pipeline, spacy_nlp, entity_emb, relation_emb,
                             ent2id, id2ent, rel2id, id2rel, ent2lbl, lbl2ent)


    demo_bot = Agent("serene-ocean", "C7qlH0L2")
    demo_bot.listen()

