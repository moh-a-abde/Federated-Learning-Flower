import pandas as pd

def clean_labeled_data(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['label'])
    drop_columns = [
        "version", "auth_attempts", "curve", "server_name", "resumed", "established", "ssl_history", "addl",
        "user_agent", "certificate.curve", "referrer", "host", "server", "status_msg", "cipher", "tags",
        "response_body_len", "status_code", "pkt_lag", "request_body_len", "uri", "service", "client", "mac_alg",
        "method", "trans_depth", "cipher_alg", "host_key", 'rtt', 'query', 'qclass', 'qclass_name', 'qtype',
        'qtype_name', 'rcode', 'rcode_name', 'AA', 'TC', 'RD', 'RA', 'Z', 'answers', 'TTLs', 'rejected',
        'compression_alg', 'kex_alg', 'host_key_alg', 'auth_success', "orig_fuids", "orig_mime_types", "origin",
        "cause", "analyzer_kind", "analyzer_name", "failure_reason", "analyzer", "next_protocol", "id",
        "hashAlgorithm", "issuerNameHash", "issuerKeyHash", "serialNumber", "certStatus", "thisUpdate",
        "nextUpdate", "version.minor3", "last_alert", "proxied", "request_type", "till", "forwardable", "renewable",
        "cookie", "security_protocol", "cert_count", "dst", "p", "tunnel_type", "status", "request.host", "request_p",
        "bound.host", "bound_p"
    ]
    data = df.drop(columns=drop_columns, errors='ignore')
    data.fillna(method='ffill', inplace=True)
    data = data.dropna()
    output_file_path = file_path.replace("_labeled.csv", "_final.csv")
    data.to_csv(output_file_path, index=False)
    print(f"Cleaned data saved to {output_file_path}")

# List of labeled CSV files to process
labeled_files = [
    "data/raw/zeek_live_export_7052024_labeled.csv",
    "data/raw/zeek_live_export_7032024_labeled.csv",
    "data/raw/zeek_live_export_7022024_labeled.csv",
    "data/raw/zeek_live_export_7012024b_labeled.csv",
    "data/raw/zeek_live_export_7012024a_labeled.csv"
]

# Loop through each labeled CSV file and clean it
for file_path in labeled_files:
    clean_labeled_data(file_path)
