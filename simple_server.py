#!/usr/bin/env python3
"""
Simple HTTP server to serve the WebSocket test client.
This avoids CORS issues with audio worklets when opening HTML files directly.
"""

import http.server
import socketserver
import webbrowser
import threading
import time
from pathlib import Path

PORT = 3000

class CustomHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Required for audio worklets
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        super().end_headers()

def start_server():
    """Start the HTTP server."""
    print(f"🌐 Starting HTTP server on http://localhost:{PORT}")
    print(f"📁 Serving files from: {Path.cwd()}")
    
    with socketserver.TCPServer(("", PORT), CustomHTTPRequestHandler) as httpd:
        print(f"✅ Server started at http://localhost:{PORT}")
        print(f"🎤 Open http://localhost:{PORT}/websocket_test.html to test WebSocket transcription")
        print("🛑 Press Ctrl+C to stop the server")
        
        # Auto-open browser after a short delay
        def open_browser():
            time.sleep(1)
            webbrowser.open(f'http://localhost:{PORT}/websocket_test.html')
        
        threading.Thread(target=open_browser, daemon=True).start()
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n🛑 Server stopped")

if __name__ == "__main__":
    start_server()
