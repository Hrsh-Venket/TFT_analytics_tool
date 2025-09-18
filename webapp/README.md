---
noteId: "d18ffbb0948911f0aa190dfc2b37a46b"
tags: []

---

# TFT Analytics HTMX Frontend

Ultra-fast, minimal TFT analytics dashboard built with HTMX and Tailwind CSS.

## Features

- **Real-time Dashboard**: Live cluster statistics and performance metrics
- **Interactive Query Builder**: Execute TFT queries with instant results
- **Cluster Explorer**: Browse meta clusters with detailed breakdowns
- **Responsive Design**: Works on desktop and mobile devices
- **Zero Build Process**: Single HTML file, ready to deploy

## Quick Start

### Local Development
```bash
# Serve locally (any HTTP server works)
python -m http.server 8000
# or
npx serve .
# or
php -S localhost:8000
```

### Firebase Hosting (Recommended)
```bash
# Install Firebase CLI
npm install -g firebase-tools

# Initialize Firebase project
firebase init hosting

# Deploy
firebase deploy
```

### Netlify (Alternative)
1. Drag the `webapp` folder to https://app.netlify.com/drop
2. Your site is live instantly

## Architecture

- **Frontend**: Single HTML file with HTMX for dynamic content
- **Styling**: Tailwind CSS via CDN with custom TFT theme
- **Backend**: Google Cloud Functions (already deployed)
- **APIs**: RESTful JSON endpoints with CORS enabled

## API Endpoints Used

- `api-stats`: Global database statistics
- `api-clusters`: Main cluster listing with metrics
- `api-cluster-details`: Detailed cluster breakdown
- `api-query`: TFT query execution engine

## Customization

### Styling
The app uses a custom TFT color palette:
- Gold: `#C89B3C` (primary accent)
- Dark Blue: `#0F1419` (background)
- Dark Gray: `#1E2328` (cards)
- Light: `#F0E6D2` (text)

### Adding Features
The modular HTMX structure makes it easy to add new sections:

```html
<!-- New section template -->
<div class="tft-card p-6">
    <h2 class="text-2xl font-bold text-tft-gold mb-6">New Feature</h2>
    <div hx-get="your-api-endpoint"
         hx-trigger="load"
         hx-target="this">
        Loading...
    </div>
</div>
```

## Performance

- **Load Time**: < 1 second (CDN resources)
- **API Calls**: Only on user interaction
- **Bundle Size**: ~15KB (single HTML file)
- **Compatibility**: Works in all modern browsers

## Deployment URLs

After deployment, you'll have a static site that connects to:
- **Cloud Functions**: `us-central1-tft-analytics-tool.cloudfunctions.net`
- **BigQuery**: Data served via serverless functions
- **Real-time**: No caching, always fresh data

Perfect for quick prototyping and production-ready analytics dashboards!