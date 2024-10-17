import rdflib
import pandas as pd
from rdflib.namespace import RDF, RDFS

# Path to your NT knowledge graph file
graph_path = '../ATAI/data/14_graph.nt'

# Load RDF graph
g = rdflib.Graph()
g.parse(graph_path, format='turtle')

# Initialize an empty list to hold the predicates and labels
predicates = []

# Iterate through all predicates in the graph
for s, p, o in g:
    # Convert the predicate (p) to a string
    predicate_str = str(p)
    
    # Query the graph for an rdfs:label
    label_list = list(g.objects(p, RDFS.label))
    
    # If a label exists, use it; otherwise, use the predicate URI
    if label_list:
        label = str(label_list[0])
    else:
        label = predicate_str.split('/')[-1]  # Fallback to the last part of the URI
    
    # Append the predicate and label to the list
    predicates.append((str(label), predicate_str))

# Convert the list of predicates to a pandas DataFrame
pred_df = pd.DataFrame(predicates, columns=["Label", "Key"])

# Remove duplicates (in case a predicate appears more than once in the RDF graph)
pred_df.drop_duplicates(inplace=True)

# Save the DataFrame to a CSV file
output_csv_path = 'word_pred.csv'
pred_df.to_csv(output_csv_path, index=False)

print(f"Predicate Lookup Table saved to {output_csv_path}")