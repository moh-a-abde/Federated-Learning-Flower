import pandas as pd

# Step 1: Read the CSV file into a DataFrame
df = pd.read_csv('data/raw/zeek_live_export_7012024a_labeled.csv')

df.head()

#ts: Timestamp when the event was logged.
#uid: Unique identifier for the connection.
#id.orig_h: Original host's IP address (the source of the connection).
#id.orig_p: Original host's port (the source port of the connection).
#id.resp_h: Responding host's IP address (the destination of the connection).
#id.resp_p: Responding host's port (the destination port of the connection).
#proto: Protocol used for the connection (e.g., TCP, UDP).
#query: Query made (applicable for DNS queries or other types of requests).
#label_service: Type of service detected in the connection (e.g., HTTP, SSH).
#label_conn_state: State of the connection (e.g., S0 for connection attempt seen,
#SYN-seen; S1 for connection established and not terminated;
#S2 for connection established and close attempt by the originator; etc.).

# Remove rows where the label is NaN
data = df.dropna(subset=['label'])

# Step 3: Save the cleaned DataFrame back to a CSV file
data.to_csv('final_form_csv', index=False)

print("NA values removed and cleaned file saved successfully.")

data.head()

# Define the columns to be dropped
drop_columns = [
    "version",	"auth_attempts", "curve"
    , "server_name", "resumed",	"established", "ssl_history",	"addl",	"user_agent",
    "certificate.curve", "referrer", "host", "server", "status_msg", "cipher", "tags",
    "response_body_len", "status_code", "pkt_lag", "request_body_len", "uri", "service",
    "client", "mac_alg", "method", "trans_depth", "cipher_alg", "host_key", 'rtt', 'query', 'qclass', 'qclass_name',
       'qtype', 'qtype_name', 'rcode', 'rcode_name', 'AA', 'TC', 'RD', 'RA',
       'Z', 'answers', 'TTLs', 'rejected', 'compression_alg', 'kex_alg', 'host_key_alg', 'auth_success',
     "orig_fuids", "orig_mime_types", "origin", "cause", "analyzer_kind", "analyzer_name", "failure_reason",
    "analyzer", "next_protocol", "id", "hashAlgorithm", "issuerNameHash", "issuerKeyHash", "serialNumber",
    "certStatus", "thisUpdate", "nextUpdate", "version.minor3", "last_alert", "proxied", "request_type",
    "till", "forwardable", "renewable", "cookie", "security_protocol", "cert_count", "dst", "p", "tunnel_type",
    "status", "request.host", "request_p", "bound.host", "bound_p"
]

# Drop the specified columns
data = data.drop(columns=drop_columns, errors='ignore')

data.columns

data.info()

#y.unique()

data.fillna(method='ffill', inplace=True)

data = data.dropna()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Convert categorical variables
categorical_features = ['id.orig_h', 'id.resp_h', 'proto', 'history', 'uid', 'conn_state']
numerical_features = ['id.orig_p', 'orig_pkts',	'orig_ip_bytes',	'resp_pkts', 'missed_bytes'
, 'local_resp', 'local_orig', 'resp_bytes', 'orig_bytes', 'duration', 'id.resp_p']

# Define the column transformer with handle_unknown='ignore' for OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Save the updated DataFrame to a new CSV file
output_file_path = 'data/raw/zeek_live_export_7012024a_final.csv'  # Adjust the path as necessary
data.to_csv(output_file_path, index=False)

print(f"CSV file saved successfully to {output_file_path}")
