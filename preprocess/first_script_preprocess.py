import pandas as pd
import numpy as np
import warnings
from sklearn.utils import shuffle

# Ignore warnings
warnings.filterwarnings("ignore")

def read_json_file(file_path):
    try:
        df = pd.read_json(file_path, lines=True)
        print("File read successfully!")
        return df
    except ValueError as ve:
        print(f"ValueError: {ve}")
        print("There is an issue with the JSON file format.")
        return None
    except FileNotFoundError as fnfe:
        print(f"FileNotFoundError: {fnfe}")
        print("The specified file path does not exist.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# Dataset
file_path = "zeek_live_export_7012024b.json" # Path to be changed
df = read_json_file(file_path)

if df is not None:
    print(df.columns)

    # Cleaning method for df (getting rid of currently unused columns)
    def clean(dfLocal):
        # Define unwanted columns
        drop_columns = [
            "peer", "metric_type", "prefix", "name", "labels",
            "label_values", "value", "mem", "pkts_proc", "events_proc", "events_queued",
            "bytes_recv", "pkts_dropped", "pkts_link", "pkts_lag", "active_tcp_conns",
            "active_udp_conns", "active_icmp_conns", "tcp_conns", "udp_conns", "icmp_conns",
            "timers", "active_timers", "files", "active_files", "dns_requests", "active_dns_requests",
            "reassem_tcp_size", "reassem_file_size", "reassem_frag_size", "reassem_unknown_size",
            "unit", "trans_id", "software_type", "version.major", "version.minor", "version.addl",
            "unparsed_version", "port_num", "port_proto", "ts_delta", "gaps", "ack", "percent_lost",
            "action", "size", "times.modified", "times.accessed", "times.created", "times.changed",
            "mode", "stratum", "poll", "precision", "root_delay", "root_disp", "ref_id", "ref_time",
            "org_time", "rec_time", "xmt_time", "num_exts", "notice", "source", "uids", "mac", "requested_addr",
            "msg_types", "host_name", "fingerprint", "certificate.version", "certificate.serial", "certificate.subject",
            "certificate.issuer", "certificate.not_valid_before", "certificate.not_valid_after", "certificate.key_alg",
            "certificate.sig_alg", "certificate.key_type", "certificate.key_length", "certificate.exponent",
            "san.dns", "basic_constraints.ca", "host_cert", "client_cert", "fuid", "depth", "analyzers", "mime_type",
            "acks", "is_orig", "seen_bytes", "total_bytes", "missing_bytes", "overflow_bytes", "timedout", "md5", "sha1",
            "extracted", "extracted_cutoff", "resp_fuids", "resp_mime_types", "cert_chain_fps", "client_cert_chain_fps",
            "subject", "issuer", "sni_matches_cert", "validation_status", "client_addr", "version.minor2", "host_p",
            "note", "msg", "sub", "src", "actions", "email_dest", "suppress_for", "direction", "level",
            "message", "location", "server_addr", "domain", "assigned_addr", "lease_time"
        ]

        # Dropping...
        dfLocal.drop(drop_columns, axis=1, inplace=True, errors='ignore')

        # Shuffle data
        dfLocal = shuffle(dfLocal)

        return dfLocal

# Information summary
def summary(dfLocal):
    print(dfLocal.columns)
    print("number of columns", len(dfLocal.columns))
    print(dfLocal.head())
    print(dfLocal.shape)
    print(dfLocal.describe())
    print(dfLocal.describe(exclude=np.number))

df = clean(df)
summary(df)

output_file_path = 'zeek_live_data_7012024b.csv'
df.to_csv(output_file_path, index=False)
print(f"CSV file saved successfully to {output_file_path}")



