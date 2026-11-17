"""
Vercel Serverless Function entrypoint for Kisan Mandi Intelligence.
Serves a responsive overview and API endpoint on Vercel,
and directs users to the full interactive Streamlit dashboard.
"""

from http.server import BaseHTTPRequestHandler
import json


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()

        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kisan Mandi Intelligence · AI Farmer Market & Price Discovery</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --primary: #1B4D3E;
            --primary-dark: #003629;
            --secondary: #2E7D32;
            --tertiary: #E65100;
            --surface: #F9FAF5;
            --card: #FFFFFF;
            --border: #E2E6DC;
            --text-primary: #141D17;
            --text-secondary: #465248;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Plus Jakarta Sans', sans-serif;
            background-color: var(--surface);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 24px 16px;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
        }
        .header {
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 32px;
            margin-bottom: 24px;
            box-shadow: 0 2px 4px rgba(20, 29, 23, 0.04);
            text-align: center;
        }
        .logo { font-size: 48px; margin-bottom: 12px; }
        h1 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 32px;
            color: var(--primary);
            margin-bottom: 8px;
            letter-spacing: -0.01em;
        }
        .tagline {
            font-size: 16px;
            color: var(--text-secondary);
            margin-bottom: 24px;
        }
        .badge {
            display: inline-block;
            background-color: #E8F5E9;
            color: #1B5E20;
            border: 1.5px solid #81C784;
            padding: 6px 14px;
            border-radius: 9999px;
            font-weight: 600;
            font-size: 13px;
            margin-bottom: 20px;
        }
        .cta-btn {
            display: inline-block;
            background-color: var(--primary);
            color: #FFFFFF;
            text-decoration: none;
            padding: 14px 28px;
            border-radius: 8px;
            font-family: 'Space Grotesk', sans-serif;
            font-weight: 600;
            font-size: 16px;
            transition: background 0.15s ease;
            box-shadow: 0 2px 6px rgba(27, 77, 62, 0.2);
        }
        .cta-btn:hover { background-color: var(--primary-dark); }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }
        .card {
            background-color: var(--card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(20, 29, 23, 0.03);
        }
        .card h3 {
            font-family: 'Space Grotesk', sans-serif;
            font-size: 18px;
            color: var(--primary);
            margin-bottom: 10px;
        }
        .card p { font-size: 14px; color: var(--text-secondary); }
        .footer {
            text-align: center;
            font-size: 13px;
            color: var(--text-secondary);
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🌾</div>
            <div class="badge">Smart India Hackathon 2026 · Problem ID: SIH26132</div>
            <h1>Kisan Mandi Intelligence</h1>
            <p class="tagline">AI-Powered Agricultural Price Discovery & Multi-Mandi Arbitrage Platform</p>
            <div style="margin-top: 10px; margin-bottom: 20px;">
                <a href="https://share.streamlit.io/deploy?repository=Nishchal-Samadhiya/kisaan&branch=main&mainModule=app.py" target="_blank" class="cta-btn">
                    🚀 Launch Interactive Dashboard (Streamlit Cloud)
                </a>
            </div>
            <div style="font-size: 13px; color: var(--text-secondary);">
                Supports <b>Wheat, Potato, Onion, and Tomato</b> across 402 mandis in Punjab, Haryana, and Uttar Pradesh.
            </div>
        </div>

        <div class="grid">
            <div class="card">
                <h3>📈 Price Forecasting</h3>
                <p>7, 15, and 30-day out-of-sample forecasting using Naive Baseline, Moving Average, ARIMA, Linear Regression, and Random Forest models.</p>
            </div>
            <div class="card">
                <h3>🎯 Sell Now vs. Wait</h3>
                <p>Actionable decision signals based on projected peak prices, calculated holding risk, and expected gain per quintal.</p>
            </div>
            <div class="card">
                <h3>🏢 Regional Arbitrage</h3>
                <p>Inter-mandi price spread rankings with an integrated diesel transport cost & net margin calculator.</p>
            </div>
            <div class="card">
                <h3>🌐 Multilingual Ready</h3>
                <p>Available in English, हिन्दी (Hindi), and ਪੰਜਾਬੀ (Punjabi) with authentic WhatsApp advisory alerts simulator.</p>
            </div>
        </div>

        <div class="footer">
            <p><b>Kisan Mandi Intelligence</b> · Open Source Agriculture AI</p>
            <p>Repository: <a href="https://github.com/Nishchal-Samadhiya/kisaan" target="_blank" style="color: var(--primary);">github.com/Nishchal-Samadhiya/kisaan</a></p>
        </div>
    </div>
</body>
</html>
"""
        self.wfile.write(html_content.encode("utf-8"))


# Aliases for Vercel detector
app = handler
application = handler
