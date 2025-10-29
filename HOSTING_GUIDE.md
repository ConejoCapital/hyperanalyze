# 🌐 Hosting Guide for HyperAnalyze

## ❌ Why Vercel Doesn't Work

**Vercel is NOT compatible with Streamlit applications.**

- **Vercel:** Designed for static sites and serverless functions (Next.js)
- **Streamlit:** Requires persistent Python server with WebSocket support
- **Result:** App crashes immediately on Vercel

---

## ✅ Recommended Hosting Solutions

### Option 1: Streamlit Community Cloud (FREE & EASIEST) ⭐

**Best for:** Quick deployment, free hosting, official support

**Pros:**
- ✅ FREE forever (public apps)
- ✅ Made specifically for Streamlit
- ✅ One-click deployment from GitHub
- ✅ Auto-deploys on git push
- ✅ No configuration needed

**Cons:**
- ❌ Apps must be public
- ❌ Resource limits (1GB RAM per app)
- ❌ Large data files (162MB) may cause issues

**How to Deploy:**
```
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select repository: ConejoCapital/hyperanalyze
5. Select branch: main
6. Main file: dashboard.py
7. Click "Deploy"
```

**Data Issue:** Your 162MB JSON files are too large for Streamlit Cloud's free tier.

---

### Option 2: Hugging Face Spaces (FREE) 🤗

**Best for:** ML/Data apps, free hosting with GPU options

**Pros:**
- ✅ FREE (public apps)
- ✅ Supports Streamlit natively
- ✅ More generous storage than Streamlit Cloud
- ✅ Can handle larger files
- ✅ Git LFS support for large files

**Cons:**
- ❌ Setup slightly more complex
- ❌ Slower cold starts

**How to Deploy:**
```
1. Go to https://huggingface.co/spaces
2. Create new Space
3. Choose "Streamlit" as SDK
4. Connect your GitHub repo or upload files
5. Add app_file: dashboard.py
```

---

### Option 3: Railway (FREE tier available) 🚂

**Best for:** Apps needing more resources, database integration

**Pros:**
- ✅ $5/month free credits
- ✅ Easy deployment from GitHub
- ✅ More resources than Streamlit Cloud
- ✅ Can add PostgreSQL/Redis easily

**Cons:**
- ❌ Free tier limited to $5/month usage
- ❌ Requires payment method on file

**How to Deploy:**
```
1. Go to https://railway.app
2. Sign in with GitHub
3. New Project → Deploy from GitHub repo
4. Select: ConejoCapital/hyperanalyze
5. Railway auto-detects Python
6. Deploy
```

---

## 📊 Handling Your Large Data Files (162MB)

### Problem
Your data files are in `.gitignore`, so they won't be deployed with your code.

### Solutions

#### Solution 1: Sample Data + User Upload (RECOMMENDED for now)

Create a small sample dataset for demos, allow users to upload their own:

**Update `dashboard.py`:**
```python
# Add at the top of sidebar
st.sidebar.header("📁 Data Source")

data_source = st.sidebar.radio(
    "Choose data source:",
    ["Use sample data (demo)", "Upload your own file"]
)

if data_source == "Upload your own file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload node_fills JSON file",
        type=['json'],
        help="Upload your Hyperliquid historical data"
    )
    
    if uploaded_file:
        # Save temporarily
        with open('temp_data.json', 'wb') as f:
            f.write(uploaded_file.read())
        data_path = 'temp_data.json'
    else:
        st.warning("Please upload a data file to continue")
        st.stop()
else:
    # Use small sample (included in repo)
    data_path = "sample_data.json"
```

---

#### Solution 2: Cloud Storage (AWS S3, Google Cloud)

Store large files separately, load at runtime:

**Store data on:**
- AWS S3 (free tier: 5GB)
- Google Cloud Storage
- Cloudflare R2 (free: 10GB/month)

**Update `data_loader.py`:**
```python
import os
import requests

class DataLoader:
    def __init__(self, data_path=None):
        # Check for cloud storage URL
        cloud_url = os.getenv('CLOUD_STORAGE_URL')
        
        if cloud_url:
            self.data_path = self._download_from_cloud(cloud_url)
        else:
            self.data_path = data_path
    
    def _download_from_cloud(self, url):
        """Download data from cloud storage."""
        local_path = 'downloaded_data.json'
        
        if not os.path.exists(local_path):
            response = requests.get(url, stream=True)
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        
        return local_path
```

**Set environment variable in hosting platform:**
```
CLOUD_STORAGE_URL=https://your-bucket.s3.amazonaws.com/node_fills.json
```

---

#### Solution 3: Git LFS (Large File Storage)

For GitHub-based hosting:

```bash
# Install Git LFS
git lfs install

# Track large files
git lfs track "Hyperliquid Data Expanded/*.json"

# Commit
git add .gitattributes
git add "Hyperliquid Data Expanded/*.json"
git commit -m "Add large data files with LFS"
git push
```

**Note:** GitHub LFS limits:
- 1GB storage free
- 1GB bandwidth/month free
- Your 162MB fits, but bandwidth is limited

---

## 🎯 Recommended Setup for Your Use Case

Since this is a **research project** → **live trading tool**:

### For Now (Research Phase):

**Option A: Local Development** (Current setup)
```bash
# Keep running locally
streamlit run dashboard.py

# Share temporarily via ngrok for demos
brew install ngrok  # if not installed
ngrok http 8501
# Gives you: https://xxxx.ngrok.io (shareable link)
```

**Option B: Sample Data Deployment**
- Create 1000-line sample
- Deploy to Streamlit Cloud or Hugging Face
- Let collaborators upload their own full datasets

### For Later (Live Trading Phase):

**AWS/GCP with WebSocket Integration**
- Full control for live data
- Real-time order book updates
- Connect to Hyperliquid WebSocket API
- Scalable infrastructure

---

## 🚀 Quick Action Plan

Let me help you deploy with sample data:

### Step 1: Create Sample Data
```bash
# Extract first 1000 lines for demo
head -n 1000 "Hyperliquid Data Expanded/node_fills_20251027_1700-1800.json" > sample_data.json
```

### Step 2: Update Dashboard
Add file upload option + use sample by default

### Step 3: Deploy
Push to GitHub → Deploy on Streamlit Cloud (free)

---

## 📝 Platform Comparison

| Platform | Free? | Streamlit Support | Data Limit | Best For |
|----------|-------|-------------------|------------|----------|
| **Vercel** | ✅ | ❌ NO | - | Static sites only |
| **Streamlit Cloud** | ✅ | ✅ Perfect | 1GB | Quick demos |
| **Hugging Face** | ✅ | ✅ Good | ~5GB | ML apps |
| **Railway** | ⚠️ $5/mo | ✅ Good | ~10GB | Production apps |
| **Render** | ✅ | ✅ Good | ~5GB | Professional |
| **AWS/GCP** | ❌ | ✅ Full control | Unlimited | Live trading |

---

## 🤔 What Would You Prefer?

1. **Keep it local** - Just use locally for now (fastest for research)
2. **Quick demo deployment** - I'll help set up sample data + file upload
3. **Full cloud deployment** - Set up cloud storage for your 162MB files
4. **Plan for live trading** - Start architecting AWS/GCP setup

Let me know and I'll help implement it! 🚀
