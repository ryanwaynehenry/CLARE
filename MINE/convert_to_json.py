import os
import csv
import json
import argparse

def load_entities(nodes_path):
    entities = []
    with open(nodes_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # use the label if present, otherwise fall back to the id
            entity = row.get('label') or row.get('id')
            if entity:
                entities.append(entity)
    # remove duplicates and sort
    return sorted(set(entities))

def load_relations_and_edge_types(edges_path):
    triples = set()
    types = set()
    with open(edges_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            src = row['source']
            rel = row['relation']
            tgt = row['target']
            triples.add((src, rel, tgt))
            types.add(rel)
    # sort the triples and prepare for JSON
    sorted_triples = sorted(triples)
    relations = [[s, r, t] for s, r, t in sorted_triples]
    edge_types = sorted(types)
    return relations, edge_types

def process_folder(folder_path):
    nodes_csv = os.path.join(folder_path, 'nodes.csv')
    edges_csv = os.path.join(folder_path, 'edges.csv')
    if not os.path.isfile(nodes_csv) or not os.path.isfile(edges_csv):
        print(f"Skipping {folder_path}: missing nodes.csv or edges.csv")
        return

    entities = load_entities(nodes_csv)
    relations, edge_types = load_relations_and_edge_types(edges_csv)

    kg = {
        'entities': entities,
        'relations': relations,
        'edges': edge_types
    }

    # use the subfolder name for the output file
    folder_name = os.path.basename(os.path.normpath(folder_path))
    output_filename = f"{folder_name}.json"
    output_path = os.path.join(folder_path, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(kg, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(entities)} entities, {len(relations)} triples, {len(edge_types)} edge types to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert CSV-based nodes and edges in each subfolder into a JSON knowledge-graph.'
    )
    parser.add_argument(
        'input_dir',
        help='path to the folder that contains subfolders with nodes.csv and edges.csv'
    )
    args = parser.parse_args()

    for name in os.listdir(args.input_dir):
        subfolder = os.path.join(args.input_dir, name)
        if os.path.isdir(subfolder):
            process_folder(subfolder)

if __name__ == '__main__':
    main()
