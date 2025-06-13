import os
import csv
import pandas as pd
from datetime import datetime, timedelta

LABELS_CSV = 'labels.csv'
FACES_DIR = 'faces'

def get_all_labels():
    people = []
    try:
        with open(LABELS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    people.append({'id': row[0], 'name': row[1]})
    except FileNotFoundError:
        pass
    return people

import pandas as pd
from datetime import datetime, timedelta

import pandas as pd
from datetime import datetime, timedelta

def generate_report_csv(period, session_gap_minutes=10):
    df = pd.read_csv('detections.csv', names=['timestamp', 'name', 'confidence'])

    # Cleanup
    df = df[df['name'].notna()]
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df = df.dropna(subset=['timestamp'])
    df = df.sort_values(by=['name', 'timestamp'])

    # Time filter
    now = datetime.now()
    if period == 'daily':
        start_time = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'weekly':
        start_time = now - timedelta(days=7)
    elif period == 'monthly':
        start_time = now - timedelta(days=30)
    else:
        return None

    df = df[df['timestamp'] >= start_time]

    # Break into sessions using time gap
    session_gap = timedelta(minutes=session_gap_minutes)
    sessions = []

    for name, group in df.groupby('name'):
        group = group.sort_values('timestamp')
        session_start = group.iloc[0]['timestamp']
        session_end = group.iloc[0]['timestamp']

        for i in range(1, len(group)):
            current_time = group.iloc[i]['timestamp']
            if current_time - session_end > session_gap:
                sessions.append({'name': name, 'check_in': session_start, 'check_out': session_end})
                session_start = current_time
            session_end = current_time

        # Add the last session
        sessions.append({'name': name, 'check_in': session_start, 'check_out': session_end})

    # Output
    report_df = pd.DataFrame(sessions)
    output_file = f'{period}_report.csv'
    report_df.to_csv(output_file, index=False)
    return output_file


def erase_person(name):
    labels = {}
    with open(LABELS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                labels[int(row[0])] = row[1]
    
    label_to_remove = None
    for k, v in labels.items():
        if v == name:
            label_to_remove = k
            break

    if label_to_remove is None:
        return False

    with open(LABELS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for k, v in labels.items():
            if k != label_to_remove:
                writer.writerow([k, v])

    for file in os.listdir(FACES_DIR):
        if file.startswith(f"{label_to_remove}_"):
            os.remove(os.path.join(FACES_DIR, file))

    return True


    # Remove from labels
    labels.pop(label_to_remove)
    with open(LABELS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for k, v in labels.items():
            writer.writerow([k, v])

    # Remove associated face images
    for file in os.listdir(FACES_DIR):
        if file.startswith(f"{label_to_remove}_"):
            os.remove(os.path.join(FACES_DIR, file))
    return True

def fix_name(old_name, new_name):
    updated = False
    rows = []
    with open('labels.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
                if row[1] == old_name:
                    row[1] = new_name
                    updated = True
                rows.append(row)
    if updated:
        with open('labels.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
    return updated

