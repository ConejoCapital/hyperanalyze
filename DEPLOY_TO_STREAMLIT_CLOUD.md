# 🚀 Deploy to Streamlit Cloud (FREE)

## ⚡ Quick Steps (5 minutes)

### 1. Go to Streamlit Cloud
Visit: **https://share.streamlit.io**

### 2. Sign In
- Click "Sign in with GitHub"
- Authorize Streamlit

### 3. Deploy New App
- Click **"New app"** button
- Fill in:
  - **Repository:** `ConejoCapital/hyperanalyze`
  - **Branch:** `main`
  - **Main file path:** `dashboard.py`
  - **App URL:** `hyperanalyze` (or choose your own)

### 4. Click "Deploy!"
- Wait 2-3 minutes for initial deployment
- Your app will be live at: `https://hyperanalyze.streamlit.app` (or your chosen URL)

---

## ✨ What People Will See

**Your public demo includes:**
- ✅ Sample dataset (1000 blocks) - works out of the box
- ✅ File uploader - visitors can test with their own data
- ✅ All 8 analysis tabs fully functional
- ✅ Mobile-friendly responsive layout

**Data options in the deployed app:**
1. **Sample data (default)** - 1000 blocks, instant load
2. **Upload file** - Users can upload their own `node_fills_*.json`
3. **Local path** - Won't work in cloud (only for local development)

---

## 📊 Your Live Demo URL

After deployment, share this link:
```
https://hyperanalyze.streamlit.app
```
(or whatever custom URL you chose)

---

## 🔄 Auto-Deploy

**Every time you push to GitHub**, Streamlit Cloud automatically:
- Detects the change
- Rebuilds your app
- Deploys the new version

No manual intervention needed! 🎉

---

## ⚙️ Optional: Advanced Settings

After initial deployment, you can configure:

### Secrets (if needed later)
- Environment variables
- API keys
- Database credentials

### Resources
- Free tier: 1GB RAM, 1 CPU core
- Usually enough for your sample data

---

## 🎯 Expected Performance

**Sample Data (1000 blocks):**
- First load: ~10-20 seconds (processing)
- Subsequent loads: ~2-3 seconds (cached)

**Uploaded Data (162MB full dataset):**
- First load: ~2-3 minutes (processing)
- Subsequent loads: ~10-15 seconds (cached)
- May hit memory limits on free tier

---

## ✅ Checklist

Before sharing your link:
- [ ] App deployed successfully
- [ ] Sample data loads and works
- [ ] All 8 tabs are functional
- [ ] File upload works (test it!)
- [ ] Charts render correctly
- [ ] Mobile view looks good

---

## 🐛 Troubleshooting

### "App is having trouble loading"
- Check build logs in Streamlit Cloud dashboard
- Usually a dependency issue - check `requirements.txt`

### "Out of memory"
- Sample data should work fine
- Full datasets may need paid tier or optimization

### "Module not found"
- Make sure all files are pushed to GitHub
- Check `requirements.txt` includes all packages

---

## 🎨 Customization

Want to change the URL or settings?
1. Go to your app dashboard on Streamlit Cloud
2. Click "⚙️ Settings"
3. Update URL, Python version, etc.

---

## 📞 Need Help?

**Streamlit Community:** https://discuss.streamlit.io  
**Docs:** https://docs.streamlit.io/streamlit-community-cloud

---

## 🌟 Result

You'll have a **public, shareable demo** at a clean URL like:

```
https://hyperanalyze.streamlit.app
```

People can:
- ✅ See your work immediately
- ✅ Try the sample data
- ✅ Upload their own Hyperliquid data
- ✅ Explore all visualizations
- ✅ Use on mobile devices

**Much better than Vercel (which doesn't work)!** 🚀

---

*Ready to deploy? Go to https://share.streamlit.io and follow the steps above!*

