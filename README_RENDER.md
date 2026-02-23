Deploying this Streamlit model on Render

Quick steps

1. Ensure your repo contains:
   - `app.py` (Streamlit app entry)
   - `requirements.txt` (project dependencies)
   - `render.yaml` (optional: service config)
   - your model `.pkl` files (e.g., `salary_model.pkl`) checked into the repo or accessible from storage

2. Correct `requirements.txt` filename if misspelled (`requirments.txt` -> `requirements.txt`).

3. Push the repository to GitHub.

4. In Render (https://render.com):
   - Create a new Web Service.
   - Connect your GitHub repo and choose a branch.
   - For "Build Command" leave blank if using `render.yaml`, otherwise use: `pip install -r requirements.txt`.
   - For "Start Command" use: 

     streamlit run app.py --server.port $PORT --server.headless true

   - Set the environment to Python and choose the instance plan.

5. Environment variables and files:
   - If your models are large, consider using an external storage (S3) and load them at runtime, or use Render's private services / deploy via Docker.
   - If you keep `.pkl` files in the repo, watch repo size limits.

6. Logs & Testing
   - After deployment, open the service URL provided by Render.
   - Check deploy logs on Render for build/runtime errors.

Local test

Run locally before pushing:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Notes

- This app uses Streamlit; you don't need Gunicorn. Use the Streamlit start command above.
- If you prefer a Docker deployment, create a `Dockerfile` and use Render's Docker service type. 
