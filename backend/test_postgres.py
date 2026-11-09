from database import pool

try:
    with pool.connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("SELECT current_database(), version()")
            result = cursor.fetchone()

            print("Connected to PostgreSQL successfully!")
            print("Database:", result[0])
            print("Version:", result[1])

finally:
    pool.close()