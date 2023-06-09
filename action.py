import csv
import psycopg2

decision_file_name = 'ai_models/decision.csv'
# Connect to your postgres server
conn = psycopg2.connect(
    dbname="gurkanindibay",
    user="gurkanindibay",
    password="",
    host="localhost",
    port="9700"
)
cur = conn.cursor()

def get_node_id(shard_id):
    cur.execute(f"SELECT n.nodeid from citus_shards c, pg_dist_node n, pg_dist_shard s \
                    WHERE c.nodeport = n.nodeport and s.shardid = c.shardid \
                    AND s.shardid = {shard_id};")
    node_id_data = cur.fetchone()
    return node_id_data

def get_key_range_for_shard(shard_id):
    # Query the pg_dist_shard table
    cur.execute(f"SELECT shardminvalue, shardmaxvalue FROM pg_dist_shard WHERE shardid = {shard_id};")
    range = cur.fetchone()
    if range is None:
        print(f"No range found for shard id {shard_id}")
        return None
    return range


def split_shard(shard_id):
    range = get_key_range_for_shard(shard_id)
    if range is None:
        print(f"No range found for shard id {shard_id}")
        return
    
    minvalue, maxvalue = int(range[0]), int(range[1])
    
    # Calculate the midpoint of the range
    midpoint = str(minvalue + (maxvalue - minvalue) // 2)
    
    node_data = get_node_id(shard_id)
    if node_data is None:
        print(f"Node not found for shard id {shard_id}")
        return
    
    node_id = node_data[0]

    # Split the shard at the midpoint
    print(f"Splitting shard: {shard_id} at midpoint: {midpoint} on node: {node_id}")
    query = f"SELECT citus_split_shard_by_split_points({shard_id}, ARRAY['{midpoint}'], ARRAY[{node_id}, {node_id}], 'block_writes');"
    print(query)
    cur.execute(query)
    conn.commit()
    print(f"Split operation of {shard_id} successful. ")
def isolate_shard(table_name, tenent_id):  
    
    # Split the shard at the midpoint
    query = f"SELECT isolate_tenant_to_new_shard('{table_name}', {tenent_id});"
    cur.execute(query)
    
    

def get_reader(file):
    for line in file:
        if not line.startswith('#'):
            yield line
                
# Open the CSV file
with open(decision_file_name, 'r') as file:
    fieldnames = file.readline().strip().lstrip('#').split(',')
    reader = csv.DictReader(file, fieldnames=fieldnames)
    for row in reader:
       if row['decision'].strip() == 'split':
            shard_id = int(row['shardid'].strip())
            split_shard(shard_id)
       if row['decision'].strip() == 'isolate':
            table_name = row['table_name'].strip()
            tenent_id = int(row['tenantid'].strip())
            isolate_shard(table_name,tenent_id)
cur.close()
conn.close()
