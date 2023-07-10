import csv
import copy
import sqlite3
import pandas as pd

def create_table_from_dict(db_file, table_name, table_dict):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Generate the CREATE TABLE statement dynamically
    columns = []
    for column_name in table_dict.keys():
        columns.append(f"{column_name} TEXT")
    create_table_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({', '.join(columns)})"

    # Execute the CREATE TABLE statement
    cursor.execute(create_table_query)

    # Commit the changes and close the connection
    conn.commit()
    conn.close()




def insert_row(db_file, table_name, row_data):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Retrieve the column names from the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]

    # Convert boolean values to integers (0 or 1)
    converted_row_data = {k: int(v) if isinstance(v, bool) else v for k, v in row_data.items()}

    # Check if all specified key-value pairs exist in the table
    query = f"SELECT COUNT(*) FROM {table_name} WHERE "
    conditions = []
    values = []
    for key, value in converted_row_data.items():
        conditions.append(f"{key} = ?")
        values.append(value)
    query += " AND ".join(conditions)
    cursor.execute(query, tuple(values))
    count = cursor.fetchone()[0]

    # If all parameters don't exist, insert a new row; otherwise, update the existing row
    if count == 0:
        placeholders = ', '.join(['?' for _ in range(len(converted_row_data))])
        insert_query = f"INSERT INTO {table_name} ({', '.join(converted_row_data.keys())}) VALUES ({placeholders})"
        cursor.execute(insert_query, tuple(converted_row_data.values()))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    
    

def update_or_insert_column(db_file, table_name, row_id, column_name, column_value):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Check if the column exists in the table
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [column[1] for column in cursor.fetchall()]
    if column_name not in columns:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name}")

    # Check if the row exists in the table
    cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE rowid = ?", (row_id,))
    count = cursor.fetchone()[0]

    # Update or insert the column value in the row
    if count == 0:
        cursor.execute(f"INSERT INTO {table_name} (rowid, {column_name}) VALUES (?, ?)", (row_id, column_value))
    else:
        cursor.execute(f"UPDATE {table_name} SET {column_name} = ? WHERE rowid = ?", (column_value, row_id))

    # Commit the changes and close the connection
    conn.commit()
    conn.close()
    

    
    
def fetch_row_by_rowid(db_file, table_name, row_id):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Fetch the row with the specified rowid
    query = f"SELECT * FROM {table_name} WHERE rowid = ?"
    cursor.execute(query, (row_id,))
    row = cursor.fetchone()

    # Get the column names
    cursor.execute(f"PRAGMA table_info({table_name})")
    column_names = [column[1] for column in cursor.fetchall()]

    # Create a dictionary with column names as keys and row values as values
    row_dict = dict(zip(column_names, row)) if row else None

    # Close the connection
    conn.close()

    return row_dict


    
def fetch_table_as_dataframe(db_file, table_name):
    # Connect to the SQLite database
    conn = sqlite3.connect(db_file)

    # Read the table into a DataFrame
    query = f"SELECT rowid,* FROM {table_name}"
    df = pd.read_sql_query(query, conn)

    # Close the connection
    conn.close()

    return df


