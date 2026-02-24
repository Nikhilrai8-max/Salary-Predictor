#!/usr/bin/env bash
set -e

# Start Streamlit with the port Render provides
streamlit run app.py --server.port "$PORT" --server.headless true
