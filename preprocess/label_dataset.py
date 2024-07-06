import pandas as pd

# Define the port-label mapping
port_label_mapping = {
    53: 'DNS',
    22: 'SSH',
    80: 'HTTP',
    443: 'HTTPS',
    21: 'FTP'
}

def label_data(file_path):
    data = pd.read_csv(file_path)
    data['label'] = data['id.resp_p'].apply(lambda x: port_label_mapping.get(int(x), None) if not pd.isna(x) else None)
    output_file_path = file_path.replace(".csv", "_labeled.csv")
    data.to_csv(output_file_path, index=False)
    print(f"Labeled data saved to {output_file_path}")
    print(data.head())

# List of CSV files to process
csv_files = [
    "data/raw/zeek_live_export_7052024.csv",
    "data/raw/zeek_live_export_7032024.csv",
    "data/raw/zeek_live_export_7022024.csv",
    "data/raw/zeek_live_export_7012024b.csv",
    "data/raw/zeek_live_export_7012024a.csv"
]

# Loop through each CSV file and label it
for file_path in csv_files:
    label_data(file_path)
