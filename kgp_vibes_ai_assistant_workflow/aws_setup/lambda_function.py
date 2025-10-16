import json
import pymysql
import os
from datetime import datetime, date
from decimal import Decimal
from urllib.parse import unquote_plus

# Custom JSON encoder to handle datetime and Decimal types
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        return super(CustomJSONEncoder, self).default(obj)

def lambda_handler(event, context):
    """
    Handle GET requests only
    Pass query as URL parameter: ?query=SELECT * FROM customers LIMIT 5
    or shortened: ?q=SELECT * FROM customers LIMIT 5
    """
    
    # Get query from URL parameters
    query_params = event.get('queryStringParameters') or {}
    
    # Support both 'query' and 'q' parameter names
    query = query_params.get('query') or query_params.get('q')
    
    if not query:
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'No query provided',
                'usage': 'Add ?query=YOUR_SQL_QUERY to the URL',
                'example': '/query?query=SELECT * FROM customers LIMIT 5'
            })
        }
    
    # URL decode the query (in case it's encoded)
    query = unquote_plus(query)
    
    # Get database credentials from environment variables
    db_config = {
        'host': os.environ.get('DB_HOST'),
        'user': os.environ.get('DB_USER'),
        'password': os.environ.get('DB_PASSWORD'),
        'database': os.environ.get('DB_NAME'),
        'port': int(os.environ.get('DB_PORT', 3306))
    }
    
    # Validate required parameters
    if not all([db_config['host'], db_config['user'], db_config['password'], db_config['database']]):
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({'error': 'Database configuration incomplete'})
        }
    
    # Connect to database and execute query
    try:
        connection = pymysql.connect(**db_config)
        
        with connection.cursor(pymysql.cursors.DictCursor) as cursor:
            # Execute query
            cursor.execute(query)
            
            # Check if it's a SELECT query
            if query.strip().upper().startswith('SELECT') or query.strip().upper().startswith('SHOW'):
                results = cursor.fetchall()
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': True,
                        'data': results,
                        'count': len(results)
                    }, cls=CustomJSONEncoder)
                }
            else:
                # For INSERT, UPDATE, DELETE
                connection.commit()
                return {
                    'statusCode': 200,
                    'headers': {
                        'Content-Type': 'application/json',
                        'Access-Control-Allow-Origin': '*'
                    },
                    'body': json.dumps({
                        'success': True,
                        'affected_rows': cursor.rowcount,
                        'last_insert_id': cursor.lastrowid
                    }, cls=CustomJSONEncoder)
                }
                
    except pymysql.Error as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Database error: {str(e)}'
            })
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': f'Unexpected error: {str(e)}'
            })
        }
    finally:
        if 'connection' in locals():
            connection.close()