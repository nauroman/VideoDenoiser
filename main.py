"""
Main entry point for Video Denoiser application
Starts the web server and opens browser
"""

import sys
import webbrowser
import time
import logging
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend import start_server

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def open_browser(url: str, delay: float = 1.5):
    """Open browser after a delay"""
    time.sleep(delay)
    logger.info(f"Opening browser at {url}")
    webbrowser.open(url)


def main():
    """Main entry point"""
    host = "127.0.0.1"
    port = 8000
    url = f"http://{host}:{port}"
    
    logger.info("=" * 60)
    logger.info("Video Denoiser - AI-Powered Video Enhancement")
    logger.info("=" * 60)
    logger.info(f"Starting server at {url}")
    logger.info("Press Ctrl+C to stop")
    logger.info("=" * 60)
    
    # Open browser in background
    import threading
    threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    
    # Start server
    try:
        start_server(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("\nShutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
