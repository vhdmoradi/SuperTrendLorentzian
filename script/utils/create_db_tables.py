import functions


def create_tables():
    db_connection, db_cursor = functions.db_connect("main_db")

    query = """
        CREATE TABLE symbols (
            id SERIAL PRIMARY KEY,
            symbol_name VARCHAR(32) NOT NULL,
            exchange VARCHAR(40) NOT NULL,
            active BOOLEAN NOT NULL
        );
        
        CREATE TABLE alerts (
            id SERIAL PRIMARY KEY,
            symbol INT,
            FOREIGN KEY (symbol) REFERENCES symbols(id),
            timeframe BIGINT NOT NULL,
            created_at TIMESTAMP NOT NULL,
            exchange VARCHAR(40) NOT NULL,
            entryexit VARCHAR(5) NOT NULL
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


create_tables()
