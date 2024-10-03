import sys
sys.path.append('/opt/homebrew/anaconda3/lib/python3.11/site-packages')

from speakeasypy import Speakeasy, Chatroom
from typing import List
import time
from rdflib import Graph, Namespace
from rdflib.namespace import RDF

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'
listen_freq = 2

# Define a namespace for your knowledge graph
EX = Namespace("http://example.org/")

# Load the RDF knowledge graph using N-Triples format
print("Loading RDF knowledge graph...")
g = Graph()
try:
    g.parse(source="../14_graph.nt", format="turtle")  # Replace with actual file and format
    print("Knowledge graph loaded successfully.")
except Exception as e:
    print(f"Failed to load the knowledge graph: {e}")
    exit(1)


class Agent:
    def __init__(self, username, password):
        self.username = username
        # Initialize the Speakeasy Python framework and login.
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.

    def execute_sparql_query(self, query):
        try:
            results = g.query(query)  # Execute the SPARQL query on the graph
            response = ""
            # Iterate through the results and build the response
            for row in results:
                response += " | ".join([str(item) for item in row]) + "\n"
            return response or "No results found."
        except Exception as e:
            return f"Error executing query: {str(e)}"

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
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Check if the message is a SPARQL query
                    if message.message.strip().lower().startswith("sparql"):
                        # Extract the SPARQL query from the message
                        sparql_query = message.message.replace("sparql", "", 1).strip()
                        # Execute the query on the graph
                        result = self.execute_sparql_query(sparql_query)
                        # Post the SPARQL query result to the room
                        room.post_messages(f"SPARQL Result:\n{result}")
                    else:
                        # Echo the non-SPARQL message back
                        room.post_messages(f"Received your message: '{message.message}' ")

                    # Mark the message as processed
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                for reaction in room.get_reactions(only_new=True):
                    print(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent("serene-ocean", "C7qlH0L2")
    demo_bot.listen()
