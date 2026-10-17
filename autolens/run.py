"""Start the API and frontend together on localhost."""
import argparse
import uvicorn

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AutoLens vehicle valuation demo')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()
    print(f'AutoLens frontend: http://{args.host}:{args.port}')
    print(f'Interactive API docs: http://{args.host}:{args.port}/docs')
    uvicorn.run('app.main:app', host=args.host, port=args.port)
