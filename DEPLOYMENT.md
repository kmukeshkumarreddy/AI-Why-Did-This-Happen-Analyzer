# Deployment Guide: AI "Why Did This Happen?" Analyzer

This guide provides step-by-step instructions to deploy the **AI "Why Did This Happen?" Analyzer** to **Render** and **Vercel**.

---

## Architecture Overview

```
                                      +------------------------------------+
                                      |            Client Browser          |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |            Vercel Edge             |
                                      |  (Global CDN, SSL & Custom Domain) |
                                      +-----------------+------------------+
                                                        |
                                                        v
                                      +------------------------------------+
                                      |          Render Web Service        |
                                      |   (Python 3.12 / Docker Container)  |
                                      |    Streamlit Reactive State Engine |
                                      |  Scikit-Learn, SHAP, LIME, XGBoost |
                                      +------------------------------------+
```

- **Render**: Hosts the stateful Python Streamlit engine with persistent bidirectional WebSockets, high memory allocation for ML models (SHAP/LIME/XGBoost), and document generation (ReportLab & DOCX).
- **Vercel**: Serves as the high-speed global Edge CDN gateway, healthcheck router, and custom domain entry point via `vercel.json`.

---

## 1. Preparing the Git Repository

Open your terminal in the project directory and run:

```bash
# 1. Initialize Git repository
git init

# 2. Stage all files (virtual environment and caches are automatically ignored by .gitignore)
git add .

# 3. Create the initial commit
git commit -m "feat: complete AI Why Did This Happen Analyzer with Render and Vercel deployment configurations"

# 4. Link your remote GitHub repository (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/nithinkumarreddy1634/AI-Why-Did-This-Happen-Analyzer.git

# 5. Push to GitHub
git push -u origin main
```

---

## 2. Deploying to Render (Recommended Host for Streamlit)

Render provides native support for long-running Streamlit processes and WebSocket communication.

### Option A: 1-Click Blueprint Deployment (render.yaml)

1. Log in to [Render Dashboard](https://dashboard.render.com).
2. Click **New +** and select **Blueprint**.
3. Connect your GitHub repository: `ai-why-did-this-happen-analyzer`.
4. Render will automatically detect the [`render.yaml`](file:///c:/AI%20Why%20Did%20This%20Happen%20Analyzer/render.yaml) blueprint file.
5. Click **Apply**. Render will automatically provision the service, install dependencies, and launch Streamlit!

---

### Option B: Manual Web Service Setup on Render

1. On the [Render Dashboard](https://dashboard.render.com), click **New +** -> **Web Service**.
2. Select your connected GitHub repository.
3. Configure the following settings:
   - **Name**: `ai-why-did-this-happen-analyzer`
   - **Region**: Choose the closest region (e.g., Oregon, Ohio, Frankfurt, Singapore)
   - **Branch**: `main`
   - **Runtime**: `Python 3`
   - **Build Command**:
     ```bash
     pip install --upgrade pip && pip install -r requirements.txt
     ```
   - **Start Command**:
     ```bash
     streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
     ```
   - **Plan**: `Free` or `Starter`
4. Expand **Advanced** -> **Add Environment Variable**:
   - `PYTHON_VERSION`: `3.12.6`
   - `STREAMLIT_SERVER_HEADLESS`: `true`
   - `STREAMLIT_SERVER_ENABLE_CORS`: `false`
   - `STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION`: `false`
5. Click **Create Web Service**.
6. When deployment finishes, your app will be live at:
   `https://ai-why-did-this-happen-analyzer.onrender.com`

---

### Option C: Docker Deployment on Render

Render also supports native container builds using our provided [`Dockerfile`](file:///c:/AI%20Why%20Did%20This%20Happen%20Analyzer/Dockerfile):

1. Click **New +** -> **Web Service**.
2. Connect your repo and set **Runtime** to `Docker`.
3. Render will build the container from `Dockerfile` and map `$PORT` automatically.

---

## 3. Deploying to Vercel

### Deploy via Vercel Dashboard

1. Log in to [Vercel](https://vercel.com).
2. Click **Add New...** -> **Project**.
3. Import your GitHub repository (`ai-why-did-this-happen-analyzer`).
4. In **Project Settings**:
   - Framework Preset: `Other`
   - Root Directory: `./`
5. Click **Deploy**.

Vercel will build the portal from [`public/index.html`](file:///c:/AI%20Why%20Did%20This%20Happen%20Analyzer/public/index.html) and deploy serverless endpoints from [`api/health.py`](file:///c:/AI%20Why%20Did%20This%20Happen%20Analyzer/api/health.py).

### Linking Vercel to your Render App

To have your Vercel URL directly proxy to your Render Streamlit application:
1. Open [`vercel.json`](file:///c:/AI%20Why%20Did%20This%20Happen%20Analyzer/vercel.json).
2. Update the destination URL with your actual Render deployment URL:
   ```json
   {
     "rewrites": [
       {
         "source": "/app/(.*)",
         "destination": "https://ai-why-did-this-happen-analyzer.onrender.com/$1"
       }
     ]
   }
   ```
3. Commit and push to GitHub. Vercel will auto-deploy the updated proxy rules!

---

## 4. Verification and Health Check

Once deployed, verify your instances:

1. **Streamlit Health Check**:
   ```
   GET https://your-render-app.onrender.com/_stcore/health
   Response: 200 OK
   ```

2. **Vercel API Gateway Health Check**:
   ```
   GET https://your-vercel-domain.vercel.app/api/health
   Response:
   {
     "status": "healthy",
     "service": "AI Why Did This Happen Analyzer - Vercel Gateway",
     "version": "1.0.0",
     "backend": "Render Streamlit Engine",
     "tests_passing": 203
   }
   ```

3. **Full Application**:
   Navigate to your Render or Vercel URL in your browser to access the interactive 10-stage root cause investigation engine.

