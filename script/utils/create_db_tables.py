import functions


def create_tables():
    db_connection, db_cursor = functions.db_connect("main_db")

    query = """
        
        CREATE TABLE alerts (
            id SERIAL PRIMARY KEY,
            symbol VARCHAR(32) NOT NULL, 
            timeframe VARCHAR(5) NOT NULL,
            created_at TIMESTAMP NOT NULL,
            exchange VARCHAR(40) NOT NULL,
            entryexit VARCHAR(5) NOT NULL,
            entry_price DOUBLE PRECISION,
            exit_price DOUBLE PRECISION,
            exit_type VARCHAR(2),
            
        );
        
        CREATE TABLE error_log (
            id SERIAL PRIMARY KEY,
            error_from VARCHAR(50) NOT NULL,
            error_text TEXT NOT NULL,
            created_at TIMESTAMP NOT NULL
        );
    """
    try:
        db_cursor.execute(query)
    except Exception as e:
        print(e)
    finally:
        db_cursor.close()
        db_connection.close()
